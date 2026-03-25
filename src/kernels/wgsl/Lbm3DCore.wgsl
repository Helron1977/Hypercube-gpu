struct Params {
    nx: u32, ny: u32, lx: u32, ly: u32,
    t: f32, tick: u32, strideFace: u32, numFaces: u32,
    p0: f32, p1: f32, p2: f32, p3: f32, p4: f32, p5: f32, p6: f32, p7: f32,
    f0: u32, f1: u32, f2: u32, f3: u32, f4: u32, f5: u32, f6: u32, f7: u32,
    f8: u32, f9: u32, f10: u32, f11: u32, f12: u32, f13: u32, f14: u32, f15: u32,
    leftRole: u32, rightRole: u32, topRole: u32, bottomRole: u32, frontRole: u32, backRole: u32
};

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

// D3Q19 Constants
const DX = array<i32, 19>(0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0);
const DY = array<i32, 19>(0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1);
const DZ = array<i32, 19>(0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1);
const OPP = array<u32, 19>(0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17);
const W  = array<f32, 19>(1.0/3.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
                         1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
                         1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0);

fn get_f_idx(d: u32, parity: u32, strideFace: u32, fBase: u32, i: u32) -> u32 {
    return (fBase + parity * 19u + d) * strideFace + i;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let nx = params.nx; let ny = params.ny; let nz = u32(params.p0); // p0: nz for 3D
    if (id.x >= nx || id.y >= ny || id.z >= nz) { return; }

    let lx = params.lx; let ly = params.ly;
    let px = id.x + 1u; let py = id.y + 1u; let pz = id.z + 1u;
    let i = (pz * ly + py) * lx + px;
    
    let strideFace = params.strideFace;
    let readParity = params.tick % 2u;
    let writeParity = (params.tick + 1u) % 2u;
    let fBase = params.f5; // f0: rho, f1: ux, f2: uy, f3: uz, f4: type, f5: pop

    // --- 0. OBSTACLE HANDLING (Simple Bounce-Back) ---
    if (data[params.f4 * strideFace + i] > 0.5) {
        for (var d = 0u; d < 19u; d = d + 1u) {
            data[get_f_idx(d, writeParity, strideFace, fBase, i)] = data[get_f_idx(OPP[d], readParity, strideFace, fBase, i)];
        }
        return;
    }

    // --- 1. BOUNDARY ROLES (Inflow/Outflow) ---
    // Inflow (Zou-He Equilibrium) on Left
    if (px == 1u && params.leftRole == 3u) {
        let ux_inf = params.p2; // p2: inlet velocity
        let rho_inf = 1.0;
        let u2_inf = 1.5 * (ux_inf * ux_inf);
        for (var d = 0u; d < 19u; d = d + 1u) {
            let cu = 3.0 * (f32(DX[d]) * ux_inf);
            data[get_f_idx(d, writeParity, strideFace, fBase, i)] = W[d] * rho_inf * (1.0 + cu + 0.5*cu*cu - u2_inf);
        }
        return;
    }
    // Outflow (Neumann) on Right
    if (px == nx && params.rightRole == 4u) {
        let n_idx = (pz * ly + py) * lx + (nx - 1u);
        for (var d = 0u; d < 19u; d = d + 1u) {
            data[get_f_idx(d, writeParity, strideFace, fBase, i)] = data[get_f_idx(d, readParity, strideFace, fBase, n_idx)];
        }
        return;
    }
    
    // --- 2. LBM CORE (Streaming & Collision) ---
    var rho: f32 = 0.0;
    var ux: f32 = 0.0; var uy: f32 = 0.0; var uz: f32 = 0.0;
    var pf = array<f32, 19>();

    for (var d = 0u; d < 19u; d = d + 1u) {
        var n_px = u32(i32(px) - DX[d]);
        var n_py = u32(i32(py) - DY[d]);
        var n_pz = u32(i32(pz) - DZ[d]);

        // Periodic/Joint/Slip Handling via parameters
        // Roles: 0: JOINT (read Halo), 1: CONTINUITY (wrap around), default: SLIP (stay at boundary)
        if (n_px == 0u) { 
            if(params.leftRole == 1u) { n_px = nx; } 
            else if(params.leftRole == 0u) { n_px = 0u; } 
            else { n_px = 1u; } 
        }
        else if (n_px == nx+1u) { 
            if(params.rightRole == 1u) { n_px = 1u; } 
            else if(params.rightRole == 0u) { n_px = nx+1u; } 
            else { n_px = nx; } 
        }
        
        if (n_py == 0u) { 
            if(params.bottomRole == 1u) { n_py = ny; } 
            else if(params.bottomRole == 0u) { n_py = 0u; } 
            else { n_py = 1u; } 
        }
        else if (n_py == ny+1u) { 
            if(params.topRole == 1u) { n_py = 1u; } 
            else if(params.topRole == 0u) { n_py = ny+1u; } 
            else { n_py = ny; } 
        }

        if (n_pz == 0u) { 
            if(params.backRole == 1u) { n_pz = nz; } 
            else if(params.backRole == 0u) { n_pz = 0u; } 
            else { n_pz = 1u; } 
        }
        else if (n_pz == nz+1u) { 
            if(params.frontRole == 1u) { n_pz = 1u; } 
            else if(params.frontRole == 0u) { n_pz = nz+1u; } 
            else { n_pz = nz; } 
        }

        let n_idx = (n_pz * ly + n_py) * lx + n_px;
        
        // Bounce-back on internal solids sensed during streaming
        if (data[params.f4 * strideFace + n_idx] > 0.5) {
            pf[d] = data[get_f_idx(OPP[d], readParity, strideFace, fBase, i)];
        } else {
            pf[d] = data[get_f_idx(d, readParity, strideFace, fBase, n_idx)];
        }
        rho += pf[d];
        ux += f32(DX[d]) * pf[d];
        uy += f32(DY[d]) * pf[d];
        uz += f32(DZ[d]) * pf[d];
    }

    ux /= rho; uy /= rho; uz /= rho;
    let omega = params.p1; // p1: relaxation factor
    let u2 = 1.5 * (ux*ux + uy*uy + uz*uz);

    for (var d = 0u; d < 19u; d = d + 1u) {
        let cu = 3.0 * (f32(DX[d])*ux + f32(DY[d])*uy + f32(DZ[d])*uz);
        let feq = W[d] * rho * (1.0 + cu + 0.5*cu*cu - u2);
        data[get_f_idx(d, writeParity, strideFace, fBase, i)] = pf[d] * (1.0 - omega) + feq * omega;
    }

    // Write Macro
    data[params.f0 * strideFace + i] = rho;
    data[params.f1 * strideFace + i] = ux;
    data[params.f2 * strideFace + i] = uy;
    data[params.f3 * strideFace + i] = uz;
}

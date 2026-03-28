
// D3Q27 Constants
const DX = array<i32, 27>(0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1);
const DY = array<i32, 27>(0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1,-1);
const DZ = array<i32, 27>(0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1, 1, 1, 1, 1,-1,-1,-1,-1);
const OPP = array<u32, 27>(0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 26, 25, 24, 23, 22, 21, 20, 19);
const W  = array<f32, 27>(8.0/27.0, 
    2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0);

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let nx = uniforms.nx; let ny = uniforms.ny; let nz = u32(uniforms.p0); // p0: nz for 3D
    if (id.x >= nx || id.y >= ny || id.z >= nz) { return; }
    let px = id.x; let py = id.y; let pz = id.z;
    
    // --- 0. OBSTACLE HANDLING (Simple Bounce-Back) ---
    if (read3D_type_Now(px, py, pz) > 0.5) {
        for (var d = 0u; d < 27u; d = d + 1u) {
            write3D_f_Next(px, py, pz, d, read3D_f_Now(px, py, pz, OPP[d]));
        }
        return;
    }

    // --- 1. BOUNDARY ROLES ---
    // Inflow (Zou-He Equilibrium) on Left
    if (px == 0u && uniforms.leftRole == 3u) {
        let ux_inf = uniforms.p2; // p2: inlet velocity
        let rho_inf = 1.0;
        let u2_inf = 1.5 * (ux_inf * ux_inf);
        for (var d = 0u; d < 27u; d = d + 1u) {
            let cu = 3.0 * (f32(DX[d]) * ux_inf);
            write3D_f_Next(px, py, pz, d, W[d] * rho_inf * (1.0 + cu + 0.5*cu*cu - u2_inf));
        }
        return;
    }
    // Outflow (Neumann) on Right
    if (px == nx - 1u && uniforms.rightRole == 4u) {
        let n_px = nx - 2u;
        for (var d = 0u; d < 27u; d = d + 1u) {
            write3D_f_Next(px, py, pz, d, read3D_f_Now(n_px, py, pz, d));
        }
        return;
    }
    
    // --- 2. LBM CORE (Streaming & Collision) ---
    var rho: f32 = 0.0;
    var ux: f32 = 0.0; var uy: f32 = 0.0; var uz: f32 = 0.0;
    var pf = array<f32, 27>();

    for (var d = 0u; d < 27u; d = d + 1u) {
        var n_px = i32(px) - DX[d];
        var n_py = i32(py) - DY[d];
        var n_pz = i32(pz) - DZ[d];

        // Roles: 1: CONTINUITY (wrap around), default: SLIP (stay at boundary)
        if (n_px < 0) { if(uniforms.leftRole == 1u) { n_px = i32(nx)-1; } else { n_px = 0; } }
        else if (n_px >= i32(nx)) { if(uniforms.rightRole == 1u) { n_px = 0; } else { n_px = i32(nx)-1; } }
        
        if (n_py < 0) { if(uniforms.bottomRole == 1u) { n_py = i32(ny)-1; } else { n_py = 0; } }
        else if (n_py >= i32(ny)) { if(uniforms.topRole == 1u) { n_py = 0; } else { n_py = i32(ny)-1; } }

        if (n_pz < 0) { if(uniforms.backRole == 1u) { n_pz = i32(nz)-1; } else { n_pz = 0; } }
        else if (n_pz >= i32(nz)) { if(uniforms.frontRole == 1u) { n_pz = 0; } else { n_pz = i32(nz)-1; } }

        // Bounce-back on internal solids sensed during streaming
        if (read3D_type_Now(u32(n_px), u32(n_py), u32(n_pz)) > 0.5) {
            pf[d] = read3D_f_Now(px, py, pz, OPP[d]);
        } else {
            pf[d] = read3D_f_Now(u32(n_px), u32(n_py), u32(n_pz), d);
        }
        rho += pf[d];
        ux += f32(DX[d]) * pf[d];
        uy += f32(DY[d]) * pf[d];
        uz += f32(DZ[d]) * pf[d];
    }

    ux /= rho; uy /= rho; uz /= rho;
    let omega = uniforms.p1; // p1: relaxation factor
    let u2 = 1.5 * (ux*ux + uy*uy + uz*uz);

    for (var d = 0u; d < 27u; d = d + 1u) {
        let cu = 3.0 * (f32(DX[d])*ux + f32(DY[d])*uy + f32(DZ[d])*uz);
        let feq = W[d] * rho * (1.0 + cu + 0.5*cu*cu - u2);
        write3D_f_Next(px, py, pz, d, pf[d] * (1.0 - omega) + feq * omega);
    }

    // Write Macros
    write3D_rho_Now(px, py, pz, rho);
    write3D_ux_Now(px, py, pz, ux);
    write3D_uy_Now(px, py, pz, uy);
    write3D_uz_Now(px, py, pz, uz);
}

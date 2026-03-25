struct GpuObject {
    pos: vec2<f32>,
    dim: vec2<f32>,
    isObstacle: f32,
    isSmoke: f32,
    objType: u32,
    _pad: u32
};

struct Params {
    nx: u32, ny: u32, lx: u32, ly: u32,
    t: f32, tick: u32, strideFace: u32, numFaces: u32,
    p0: f32, p1: f32, p2: f32, p3: f32, p4: f32, p5: f32, p6: f32, p7: f32,
    f0: u32, f1: u32, f2: u32, f3: u32, f4: u32, f5: u32, f6: u32, f7: u32,
    f8: u32, f9: u32, f10: u32, f11: u32, f12: u32, f13: u32, f14: u32, f15: u32,
    leftRole: u32, rightRole: u32, topRole: u32, bottomRole: u32, frontRole: u32, backRole: u32,
    _pad: u32, _pad2: u32,
    objects: array<GpuObject, 8>
};

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

const DX = array<i32, 9>(0, 1, 0, -1, 0, 1, -1, -1, 1);
const DY = array<i32, 9>(0, 0, 1, 0, -1, 1, 1, -1, -1);
const W  = array<f32, 9>(4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0);
const OPP = array<u32, 9>(0, 3, 4, 1, 2, 7, 8, 5, 6);

fn lbm_feq(rho: f32, ux: f32, uy: f32, d: u32) -> f32 {
    let cu = 3.0 * (f32(DX[d]) * ux + f32(DY[d]) * uy);
    let u2 = 1.5 * (ux * ux + uy * uy);
    return W[d] * rho * (1.0 + cu + 0.5 * cu * cu - u2);
}

fn get_f_idx(d: u32, parity: u32, strideFace: u32, fBase: u32, i: u32) -> u32 {
    // MasterBuffer entrelace les buffers ping/pong : f0_ping, f0_pong, f1_ping, f1_pong...
    return (fBase + d * 2u + parity) * strideFace + i;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let nx = params.nx; let ny = params.ny;
    let lx = params.lx; let ly = params.ly;
    let strideFace = params.strideFace;
    let readParity = params.tick % 2u;
    let writeParity = (params.tick + 1u) % 2u;
    let fBase = params.f5;

    if (id.x >= nx || id.y >= ny) { return; }
    let px = id.x + 1u; let py = id.y + 1u;
    let i = py * lx + px;

    // --- STEP 0: AUTOMATIC INITIALIZATION ---
    if (params.tick == 0u) {
        let ux_init = params.p1;
        for (var d = 0u; d < 9u; d = d + 1u) {
            let feq = lbm_feq(1.0, ux_init, 0.0, d);
            data[get_f_idx(d, 0u, strideFace, fBase, i)] = feq;
            data[get_f_idx(d, 1u, strideFace, fBase, i)] = feq;
        }
        data[params.f1 * strideFace + i] = ux_init;
        data[params.f2 * strideFace + i] = 0.0;
        data[params.f3 * strideFace + i] = 1.0;
        return;
    }

    // --- 1. OBSTACLE HANDLING (Bounce-Back) ---
    if (data[params.f0 * strideFace + i] > 0.99) {
        data[params.f1 * strideFace + i] = 0.0;
        data[params.f2 * strideFace + i] = 0.0;
        for (var d = 0u; d < 9u; d = d + 1u) {
            let opp_d = OPP[d];
            data[get_f_idx(d, writeParity, strideFace, fBase, i)] = data[get_f_idx(opp_d, readParity, strideFace, fBase, i)];
        }
        return;
    }

    // --- 2. INFLOW (Zou-He / Force Equilibrium) ---
    if (px == 1u && params.leftRole == 3u) { // 3: INFLOW
        var scale = 1.0;
        if (id.y < 16u) { scale = f32(id.y) / 16.0; }
        if (id.y > ny - 17u) { scale = f32(ny - 1u - id.y) / 16.0; }
        
        let ux_inf = params.p1 * scale;
        let rho_inf = 1.0;
        let u2_inf = 1.5 * (ux_inf * ux_inf);
        
        for (var d = 0u; d < 9u; d = d + 1u) {
            let cu = 3.0 * (f32(DX[d]) * ux_inf);
            data[get_f_idx(d, writeParity, strideFace, fBase, i)] = W[d] * rho_inf * (1.0 + cu + 0.5 * cu * cu - u2_inf);
        }
        data[params.f1 * strideFace + i] = ux_inf;
        data[params.f2 * strideFace + i] = 0.0;
        data[params.f3 * strideFace + i] = 1.0;
        return;
    }

    // --- 3. OUTFLOW (Neumann) ---
    if (px == nx && params.rightRole == 4u) {
        let p_idx = py * lx + (nx - 1u);
        for (var d = 0u; d < 9u; d = d + 1u) {
            data[get_f_idx(d, writeParity, strideFace, fBase, i)] = data[get_f_idx(d, readParity, strideFace, fBase, p_idx)];
        }
        data[params.f1 * strideFace + i] = data[params.f1 * strideFace + p_idx];
        data[params.f2 * strideFace + i] = data[params.f2 * strideFace + p_idx];
        data[params.f3 * strideFace + i] = data[params.f3 * strideFace + p_idx];
        return;
    }

    // --- 4. LBM CORE (Streaming & Collision) ---
    var rho: f32 = 0.0;
    var pf = array<f32, 9>();
    for (var d = 0u; d < 9u; d = d + 1u) {
        var npx = u32(i32(px) - DX[d]);
        var npy = u32(i32(py) - DY[d]);
        
        // Enforce boundary roles dynamically on the streaming lookup vector
        if (npx == 0u) {
            if (params.leftRole == 1u) { npx = nx; }
        } else if (npx == nx + 1u) {
            if (params.rightRole == 1u) { npx = 1u; }
        }
        
        if (npy == 0u) {
            if (params.bottomRole == 1u) { npy = ny; }
        } else if (npy == ny + 1u) {
            if (params.topRole == 1u) { npy = 1u; }
        }

        let ni = npy * lx + npx;
        
        // --- BOUNDARY & OBSTACLE HANDLING ---
        // 1. Boundary Roles
        if (npx == 0u || npx == nx + 1u || npy == 0u || npy == ny + 1u) {
            var hitWall = false;
            if (npx == 0u      && params.leftRole == 2u)   { hitWall = true; }
            if (npx == nx + 1u && params.rightRole == 2u)  { hitWall = true; }
            if (npy == 0u      && params.bottomRole == 2u) { hitWall = true; }
            if (npy == ny + 1u && params.topRole == 2u)    { hitWall = true; }

            if (hitWall) {
                // Stationary Bounce-Back
                pf[d] = data[get_f_idx(OPP[d], readParity, strideFace, fBase, i)];
            } else if (npy == ny + 1u && params.topRole == 10u) {
                // MOVING WALL (Top) - Momentum adjusted bounce-back
                let u_wall = params.p2; 
                let opp_d = OPP[d];
                let dot_u = f32(DX[d]) * u_wall; 
                pf[d] = data[get_f_idx(opp_d, readParity, strideFace, fBase, i)] + 6.0 * W[d] * 1.0 * dot_u;
            } else {
                // Fallback for custom or unhandled roles (read ghost)
                pf[d] = data[get_f_idx(d, readParity, strideFace, fBase, ni)];
            }
        } 
        // 2. Internal Obstacles
        else if (data[params.f0 * strideFace + ni] > 0.99) {
            pf[d] = data[get_f_idx(OPP[d], readParity, strideFace, fBase, i)];
        } 
        // 3. Pure Fluid Streaming
        else {
            pf[d] = data[get_f_idx(d, readParity, strideFace, fBase, ni)];
        }
        
        // --- 4. SOURCE TERM (Momentum Injection) ---
        // p3: force_x, p4: force_y
        pf[d] += 3.0 * W[d] * (f32(DX[d]) * params.p3 + f32(DY[d]) * params.p4);

        rho += pf[d];
    }

    let invRho = 1.0 / rho;
    let ux = ((pf[1] + pf[5] + pf[8]) - (pf[3] + pf[6] + pf[7])) * invRho;
    let uy = ((pf[2] + pf[5] + pf[6]) - (pf[4] + pf[7] + pf[8])) * invRho;

    data[params.f1 * strideFace + i] = ux;
    data[params.f2 * strideFace + i] = uy;
    data[params.f3 * strideFace + i] = rho;

    let omega = params.p0;
    for (var d = 0u; d < 9u; d = d + 1u) {
        let feq = lbm_feq(rho, ux, uy, d);
        data[get_f_idx(d, writeParity, strideFace, fBase, i)] = pf[d] * (1.0 - omega) + feq * omega;
    }
}

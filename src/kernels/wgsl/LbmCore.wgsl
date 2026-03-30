
// Hypercube D2Q9 LBM CORE (v6.0 Alpha)
// ------------------------------------
// Optimized for Zero-Stall performance with Boundary Roles.

const DX = array<i32, 9>(0, 1, 0, -1, 0, 1, -1, -1, 1);
const DY = array<i32, 9>(0, 0, 1, 0, -1, 1, 1, -1, -1);
const W  = array<f32, 9>(4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0);
const OPP = array<u32, 9>(0, 3, 4, 1, 2, 7, 8, 5, 6);

fn lbm_feq(rho: f32, ux: f32, uy: f32, d: u32) -> f32 {
    let cu = 3.0 * (f32(DX[d]) * ux + f32(DY[d]) * uy);
    let u2 = 1.5 * (ux * ux + uy * uy);
    return W[d] * rho * (1.0 + cu + 0.5 * cu * cu - u2);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let nx = uniforms.nx; let ny = uniforms.ny;
    if (id.x >= nx || id.y >= ny) { return; }
    let px = id.x; let py = id.y;

    // --- 0. INITIALIZATION (Absolute Stability) ---
    if (uniforms.tick == 0u) {
        let ux_init = uniforms.p1;
        for (var d = 0u; d < 9u; d = d + 1u) {
            let feq = lbm_feq(1.0, ux_init, 0.0, d);
            write_f_Now(px, py, d, feq);
            write_f_Next(px, py, d, feq);
        }
        write_ux_Now(px, py, ux_init);
        write_ux_Next(px, py, ux_init);
        write_rho_Now(px, py, 1.0);
        write_rho_Next(px, py, 1.0);
        return;
    }

    // --- 1. OBSTACLES (Internal Bounce-Back) ---
    if (read_obs_Now(px, py) > 0.99) {
        for (var d = 0u; d < 9u; d = d + 1u) {
            write_f_Next(px, py, d, read_f_Now(px, py, OPP[d]));
        }
        write_ux_Next(px, py, 0.0);
        write_uy_Next(px, py, 0.0);
        return;
    }

    // --- 2. BOUNDARY ROLES ---
    // Inflow (Left)
    if (px == 0u && uniforms.leftRole == 3u) {
        let ux_in = uniforms.p1; let rho_in = 1.0;
        for (var d = 0u; d < 9u; d = d + 1u) {
            write_f_Next(px, py, d, lbm_feq(rho_in, ux_in, 0.0, d));
        }
        return;
    }
    // Outflow (Right)
    if (px == nx - 1u && uniforms.rightRole == 4u) {
        for (var d = 0u; d < 9u; d = d + 1u) {
            write_f_Next(px, py, d, read_f_Now(nx - 2u, py, d));
        }
        return;
    }

    // --- 3. STREAMING & COLLISION ---
    var pf = array<f32, 9>();
    var rho: f32 = 0.0;
    var momX: f32 = 0.0;
    var momY: f32 = 0.0;

    for (var d = 0u; d < 9u; d = d + 1u) {
        var npx = i32(px) - DX[d];
        var npy = i32(py) - DY[d];

        // Horizontal Role (Left/Right) - Rule 1: Periodic (Continuity)
        if (npx < 0) { if(uniforms.leftRole == 1u) { npx = i32(nx)-1; } else { npx = 0; } }
        else if (npx >= i32(nx)) { if(uniforms.rightRole == 1u) { npx = 0; } else { npx = i32(nx)-1; } }
        
        // Vertical Role (Bottom/Top) - Rule 1: Periodic (Continuity)
        var isWall = false;
        if (npy < 0) { 
            if(uniforms.bottomRole == 1u) { npy = i32(ny)-1; } 
            else { isWall = true; } // No-Slip Wall
        } else if (npy >= i32(ny)) { 
            if(uniforms.topRole == 1u) { npy = 0; } 
            else { isWall = true; } // No-Slip Wall
        }

        if (isWall) {
            pf[d] = read_f_Now(px, py, OPP[d]);
        } else if (read_obs_Now(u32(npx), u32(npy)) > 0.99) {
            pf[d] = read_f_Now(px, py, OPP[d]);
        } else {
            pf[d] = read_f_Now(u32(npx), u32(npy), d);
        }

        rho += pf[d];
        momX += pf[d] * f32(DX[d]);
        momY += pf[d] * f32(DY[d]);
    }

    let ux = momX / rho;
    let uy = momY / rho;
    let omega = uniforms.p0;

    for (var d = 0u; d < 9u; d = d + 1u) {
        let feq = lbm_feq(rho, ux, uy, d);
        write_f_Next(px, py, d, pf[d] + omega * (feq - pf[d]));
    }

    write_rho_Next(px, py, rho);
    write_ux_Next(px, py, ux);
    write_uy_Next(px, py, uy);

    // Diagnostics (Curl for Vorticity)
    let ux_t = read_ux_Now(px, clamp(py + 1u, 0u, ny - 1u));
    let ux_b = read_ux_Now(px, clamp(py - 1u, 0u, ny - 1u));
    let uy_r = read_uy_Now(clamp(px + 1u, 0u, nx - 1u), py);
    let uy_l = read_uy_Now(clamp(px - 1u, 0u, nx - 1u), py);
    write_curl_Next(px, py, (uy_r - uy_l) - (ux_t - ux_b));
}


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
    // Dans le layout Hypercube, Ping/Pong sont groupés par face.
    // Pour une face à 9 composants : Slots [Ping: 0..8, Pong: 9..17]
    return (fBase + (parity * 9u) + d) * strideFace + i;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {    let nx = uniforms.nx; let ny = uniforms.ny;
    
    if (id.x >= nx || id.y >= ny) { return; }
    let px = id.x; let py = id.y;

    // --- STEP 0: AUTOMATIC INITIALIZATION ---
    if (uniforms.tick == 0u) {
        let ux_init = uniforms.p1;
        for (var d = 0u; d < 9u; d = d + 1u) {
            let feq = lbm_feq(1.0, ux_init, 0.0, d);
            write_f_Now(px, py, d, feq); // Initial fill in both slots
            write_f_Next(px, py, d, feq);
        }
        write_ux_Now(px, py, ux_init);
        write_uy_Now(px, py, 0.0);
        write_rho_Now(px, py, 1.0);
        return;
    }

    // --- 1. OBSTACLE HANDLING (Bounce-Back) ---
    if (read_obs_Now(px, py) > 0.99) {
        write_ux_Now(px, py, 0.0);
        write_uy_Now(px, py, 0.0);
        for (var d = 0u; d < 9u; d = d + 1u) {
            write_f_Next(px, py, d, read_f_Now(px, py, OPP[d]));
        }
        return;
    }

    // --- 2. INFLOW (Zou-He / Force Equilibrium) ---
    if (px == 0u && uniforms.leftRole == 3u) { // 3: INFLOW
        var scale = 1.0;
        if (py < 16u) { scale = f32(py) / 16.0; }
        if (py > ny - 17u) { scale = f32(ny - 1u - py) / 16.0; }
        
        let ux_inf = uniforms.p1 * scale;
        let rho_inf = 1.0;
        let u2_inf = 1.5 * (ux_inf * ux_inf);
        
        for (var d = 0u; d < 9u; d = d + 1u) {
            let cu = 3.0 * (f32(DX[d]) * ux_inf);
            write_f_Next(px, py, d, W[d] * rho_inf * (1.0 + cu + 0.5 * cu * cu - u2_inf));
        }
        write_ux_Now(px, py, ux_inf);
        write_uy_Now(px, py, 0.0);
        write_rho_Now(px, py, 1.0);
        return;
    }

    // --- 3. OUTFLOW (Neumann) ---
    if (px == nx - 1u && uniforms.rightRole == 4u) {
        let n_px = nx - 2u;
        for (var d = 0u; d < 9u; d = d + 1u) {
            write_f_Next(px, py, d, read_f_Now(n_px, py, d));
        }
        write_ux_Now(px, py, read_ux_Now(n_px, py));
        write_uy_Now(px, py, read_uy_Now(n_px, py));
        write_rho_Now(px, py, read_rho_Now(n_px, py));
        return;
    }

    // --- 4. LBM CORE (Streaming & Collision) ---
    var rho: f32 = 0.0;
    var pf = array<f32, 9>();
    for (var d = 0u; d < 9u; d = d + 1u) {
        var npx = i32(px) - DX[d];
        var npy = i32(py) - DY[d];
        
        // Enforce boundary roles dynamically
        if (npx < 0) {
            if (uniforms.leftRole == 1u) { npx = i32(nx) - 1; }
        } else if (npx >= i32(nx)) {
            if (uniforms.rightRole == 1u) { npx = 0; }
        }
        
        if (npy < 0) {
            if (uniforms.bottomRole == 1u) { npy = i32(ny) - 1; }
        } else if (npy >= i32(ny)) {
            if (uniforms.topRole == 1u) { npy = 0; }
        }

        // --- BOUNDARY & OBSTACLE HANDLING ---
        if (npx < 0 || npx >= i32(nx) || npy < 0 || npy >= i32(ny)) {
            var hitWall = false;
            if (npx < 0           && uniforms.leftRole == 2u)   { hitWall = true; }
            if (npx >= i32(nx)    && uniforms.rightRole == 2u)  { hitWall = true; }
            if (npy < 0           && uniforms.bottomRole == 2u) { hitWall = true; }
            if (npy >= i32(ny)    && uniforms.topRole == 2u)    { hitWall = true; }

            if (hitWall) {
                pf[d] = read_f_Now(px, py, OPP[d]);
            } else if (npy >= i32(ny) && uniforms.topRole == 10u) {
                let u_wall = uniforms.p2; 
                pf[d] = read_f_Now(px, py, OPP[d]) + 6.0 * W[d] * 1.0 * (f32(DX[d]) * u_wall);
            } else {
                // Fallback (read current coord as it's at boundary)
                pf[d] = read_f_Now(px, py, d);
            }
        } 
        else if (read_obs_Now(u32(npx), u32(npy)) > 0.99) {
            pf[d] = read_f_Now(px, py, OPP[d]);
        } 
        else {
            pf[d] = read_f_Now(u32(npx), u32(npy), d);
        }
        
        // --- 4. SOURCE TERM ---
        pf[d] += 3.0 * W[d] * (f32(DX[d]) * uniforms.p3 + f32(DY[d]) * uniforms.p4);
        rho += pf[d];
    }

    let invRho = 1.0 / rho;
    let ux = ((pf[1] + pf[5] + pf[8]) - (pf[3] + pf[6] + pf[7])) * invRho;
    let uy = ((pf[2] + pf[5] + pf[6]) - (pf[4] + pf[7] + pf[8])) * invRho;

    write_ux_Now(px, py, ux);
    write_uy_Now(px, py, uy);
    write_rho_Now(px, py, rho);

    // --- 5. VORTICITY ---
    let ux_top = read_ux_Now(px, clamp(py + 1u, 0u, ny - 1u));
    let ux_bot = read_ux_Now(px, clamp(py - 1u, 0u, ny - 1u));
    let uy_rit = read_uy_Now(clamp(px + 1u, 0u, nx - 1u), py);
    let uy_lef = read_uy_Now(clamp(px - 1u, 0u, nx - 1u), py);
    write_curl_Now(px, py, (uy_rit - uy_lef) - (ux_top - ux_bot));

    let omega = uniforms.p0;
    for (var d = 0u; d < 9u; d = d + 1u) {
        let feq = lbm_feq(rho, ux, uy, d);
        write_f_Next(px, py, d, pf[d] * (1.0 - omega) + feq * omega);
    }
}

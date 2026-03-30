@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let px = id.x;
    let py = id.y;
    let nx = uniforms.nx;
    let ny = uniforms.ny;

    if (px < nx && py < ny) {
        let dt_eps = uniforms.p0;
        
        // v6.0: Explicit temporal nomenclature (_Now)
        let ez = read_Ez_Now(px, py);
        let hx = read_Hx_Now(px, py);
        let hy = read_Hy_Now(px, py);
        
        // Hybrid Maxwell E-Update
        // E += dt/eps * curl(H)
        var hx_prev = 0.0;
        if (py > 0u) { hx_prev = read_Hx_Now(px, py - 1u); }
        
        var hy_prev = 0.0;
        if (px > 0u) { hy_prev = read_Hy_Now(px - 1u, py); }
        
        let new_ez = ez + dt_eps * ((hy - hy_prev) - (hx - hx_prev));
        
        // Sine source at p3,p4
        let srcX = u32(uniforms.p3);
        let srcY = u32(uniforms.p4);
        var final_ez = new_ez;
        if (px == srcX && py == srcY) {
            final_ez += uniforms.p5;
        }

        write_Ez_Now(px, py, final_ez);
    }
}

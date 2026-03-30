@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let px = id.x;
    let py = id.y;
    let nx = uniforms.nx;
    let ny = uniforms.ny;

    if (px < nx && py < ny) {
        // v6.0: Explicit temporal nomenclature (_Now)
        // Vorticity = du/dy - dv/dx
        
        var u_next_y = 0.0;
        var u_prev_y = 0.0;
        if (py < ny - 1u) { u_next_y = read_ux_Now(px, py + 1u); }
        if (py > 0u) { u_prev_y = read_ux_Now(px, py - 1u); }
        
        var v_next_x = 0.0;
        var v_prev_x = 0.0;
        if (px < nx - 1u) { v_next_x = read_uy_Now(px + 1u, py); }
        if (px > 0u) { v_prev_x = read_uy_Now(px - 1u, py); }
        
        let curl = (u_next_y - u_prev_y) - (v_next_x - v_prev_x);
        
        write_curl_Now(px, py, curl);
    }
}

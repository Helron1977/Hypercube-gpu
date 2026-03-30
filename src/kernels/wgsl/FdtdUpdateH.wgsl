@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let px = id.x;
    let py = id.y;
    let nx = uniforms.nx;
    let ny = uniforms.ny;

    if (px < nx && py < ny) {
        let dt_mu = uniforms.p1;
        
        // v6.0: Explicit temporal nomenclature (_Now)
        let ez = read_Ez_Now(px, py);
        let hx = read_Hx_Now(px, py);
        let hy = read_Hy_Now(px, py);
        
        // Hx Update
        var ez_next_y = 0.0;
        if (py < ny - 1u) { ez_next_y = read_Ez_Now(px, py + 1u); }
        let new_hx = hx - dt_mu * (ez_next_y - ez);
        
        // Hy Update
        var ez_next_x = 0.0;
        if (px < nx - 1u) { ez_next_x = read_Ez_Now(px + 1u, py); }
        let new_hy = hy + dt_mu * (ez_next_x - ez);

        write_Hx_Now(px, py, new_hx);
        write_Hy_Now(px, py, new_hy);
    }
}

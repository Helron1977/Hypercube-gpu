// Redefinition struct Params removed - using inherited uniforms

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let nx = uniforms.nx; let ny = uniforms.ny;
    let lx = uniforms.strideRow;
    let strideFace = uniforms.strideFace;

    let px = id.x; let py = id.y;
    if (px >= nx || py >= ny) { return; }

    // TM-Mode: Ez is updated by Hx and Hy
    // Ez += dt * ((Hy - Hy_left) - (Hx - Hx_down))
    let ez = read_Ez(px, py);
    let hx = read_Hx(px, py);
    let hx_dn = read_Hx(px, py - 1u);
    let hy = read_Hy(px, py);
    let hy_lt = read_Hy(px - 1u, py);

    let dt = uniforms.p0;
    
    // Dipole Source in Center
    let isSrc = f32(nx/2u == px && ny/2u == py);
    let srcVal = isSrc * sin(f32(uniforms.tick) * 0.2) * 5.0;
    
    write_Ez(px, py, ez + dt * ((hy - hy_lt) - (hx - hx_dn)) + srcVal);
}

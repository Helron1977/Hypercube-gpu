// Redefinition struct Params removed - using inherited uniforms

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let nx = uniforms.nx; let ny = uniforms.ny;
    let lx = uniforms.strideRow;
    let strideFace = uniforms.strideFace;

    let px = id.x; let py = id.y;
    if (px >= nx || py >= ny) { return; }

    // TM-Mode: Hx and Hy are updated by Ez gradients
    // Hx -= dt * (Ez_up - Ez)
    // Hy += dt * (Ez_right - Ez)
    let hx = read_Hx(px, py);
    let hy = read_Hy(px, py);
    let ez = read_Ez(px, py);
    let ez_up = read_Ez(px, py + 1u);
    let ez_rt = read_Ez(px + 1u, py);

    let dt = uniforms.p1; // dt_mu
    
    write_Hx(px, py, hx - dt * (ez_up - ez));
    write_Hy(px, py, hy + dt * (ez_rt - ez));
}

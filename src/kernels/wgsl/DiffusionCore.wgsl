// Redefinition struct Params removed - using inherited uniforms

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let nx = uniforms.nx; let ny = uniforms.ny;
    let lx = uniforms.strideRow;
    let strideFace = uniforms.strideFace;

    if (id.x >= nx || id.y >= ny) { return; }
    let px = id.x; let py = id.y;
    let i = getIndex(px, py);

    // Alpha: p0, faces: f0=U_now, f1=U_next
    let alpha = uniforms.p0;
    
    // Diffusion stencil using Standard Macros (Host-handled parity)
    let u = read_t(px, py);
    let uL = read_t(px - 1u, py);
    let uR = read_t(px + 1u, py);
    let uT = read_t(px, py + 1u);
    let uB = read_t(px, py - 1u);
    
    let lap = (uL + uR + uT + uB - 4.0 * u);
    write_t(px, py, u + alpha * lap);
}

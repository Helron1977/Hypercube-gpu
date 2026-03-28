@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let px = id.x; let py = id.y;
    if (px >= uniforms.nx || py >= uniforms.ny) { return; }
    let i = getIndex(px, py);

    // Evaluate function: f0 = uniforms.faces[0]
    // p0: scale, p1: freq
    let x = (f32(px) / f32(uniforms.nx) - 0.5) * uniforms.p0;
    let y = (f32(py) / f32(uniforms.ny) - 0.5) * uniforms.p0;
    
    // p2 = time multiplier (0.0 = static, >0 = animated). Default 0.
    let val = sin(x * uniforms.p1 + uniforms.t * uniforms.p2) * cos(y * uniforms.p1);
    data[uniforms.faces[0] * uniforms.strideFace + i] = val;
}

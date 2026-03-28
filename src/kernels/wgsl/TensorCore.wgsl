// Redefinition struct Params removed - using inherited uniforms

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let px = id.x; let py = id.y;
    if (px >= uniforms.nx || py >= uniforms.ny) { return; }

    let i = (py + 1u) * uniforms.strideRow + (px + 1u);
    let strideFace = uniforms.strideFace;

    // Tensor Operation: Generic Map (Alpha*A + Beta*B)
    // f0: Out, f1: A, f2: B
    // p0: Alpha, p1: Beta
    let valA = data[uniforms.faces[1] * strideFace + i];
    let valB = data[uniforms.faces[2] * strideFace + i];
    
    data[uniforms.faces[0] * strideFace + i] = uniforms.p0 * valA + uniforms.p1 * valB;
}

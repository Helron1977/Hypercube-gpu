// Redefinition struct Params removed - using inherited uniforms

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let px = id.x; let py = id.y;
    if (px >= uniforms.nx || py >= uniforms.ny) { return; }

    let i = (py + 1u) * uniforms.strideRow + (px + 1u);
    
    let x = (f32(px)/f32(uniforms.nx));
    let y = (f32(py)/f32(uniforms.ny));
    
    let val = fract(sin(dot(vec2<f32>(x, y), vec2<f32>(12.9898, 78.233))) * 43758.5453 + uniforms.t);
    data[uniforms.faces[0] * uniforms.strideFace + i] = val;
}

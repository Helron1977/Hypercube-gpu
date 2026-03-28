// Redefinition struct Params removed - using inherited uniforms

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let nx = uniforms.nx; let ny = uniforms.ny;
    let lx = uniforms.strideRow;
    let strideFace = uniforms.strideFace;

    if (id.x >= nx || id.y >= ny) { return; }
    let px = id.x; let py = id.y;
    let i = getIndex(px, py);

    let f_tensor = uniforms.faces[0];
    let f_grad = uniforms.faces[1];
    
    let lr = uniforms.p1;
    let val = data[f_tensor * strideFace + i];
    let grad = data[f_grad * strideFace + i];
    
    data[f_tensor * strideFace + i] = val - lr * grad;
}

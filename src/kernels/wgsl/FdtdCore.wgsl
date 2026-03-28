// Redefinition struct Params removed - using inherited uniforms

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let nx = uniforms.nx; let ny = uniforms.ny;
    let lx = uniforms.strideRow;
    let strideFace = uniforms.strideFace;

    if (id.x >= nx || id.y >= ny) { return; }
    let px = id.x; let py = id.y;
    let i = getIndex(px, py);

    // Update rules for electromagnetic fields
    // Ex, Ey, Hz faces.
    let f_ex = uniforms.faces[0];
    let f_ey = uniforms.faces[1];
    let f_hz = uniforms.faces[2];

    // Hz update
    let hz = data[f_hz * strideFace + i];
    let ex = data[f_ex * strideFace + i];
    let ex_up = data[f_ex * strideFace + getIndex(px, py + 1u)];
    let ey = data[f_ey * strideFace + i];
    let ey_rt = data[f_ey * strideFace + getIndex(px + 1u, py)];
    
    data[f_hz * strideFace + i] = hz - uniforms.p0 * ((ey_rt - ey) - (ex_up - ex));
}

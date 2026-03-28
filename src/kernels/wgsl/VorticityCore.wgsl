// Redefinition struct Params removed - using inherited uniforms

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let nx = uniforms.nx; let ny = uniforms.ny;
    let lx = uniforms.strideRow;
    let strideFace = uniforms.strideFace;

    if (id.x >= nx || id.y >= ny) { return; }
    let px = id.x; let py = id.y;
    
    // We use getIndex for safety although manual index is fine
    let i = getIndex(px, py);

    // f1: ux, f2: uy, f4: curl
    let f_ux = uniforms.faces[1];
    let f_uy = uniforms.faces[2];
    let f_curl = uniforms.faces[4];

    if (f_curl < uniforms.numFaces) {
        let vx_up = data[f_ux * strideFace + getIndex(px, py + 1u)];
        let vx_dn = data[f_ux * strideFace + getIndex(px, py - 1u)];
        let vy_rt = data[f_uy * strideFace + getIndex(px + 1u, py)];
        let vy_lt = data[f_uy * strideFace + getIndex(px - 1u, py)];
        
        // Finite difference vorticity: (dVy/dx - dVx/dy)
        data[f_curl * strideFace + i] = (vy_rt - vy_lt) * 0.5 - (vx_up - vx_dn) * 0.5;
    }
}

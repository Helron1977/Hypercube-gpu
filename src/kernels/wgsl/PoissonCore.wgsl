// Redefinition struct Params removed - using inherited uniforms

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let px = id.x; let py = id.y;
    if (px >= uniforms.nx || py >= uniforms.ny) { return; }

    // Use generated macros (phi is first face in rule: read, write, rhs)
    let lapSum = read_phi(px - 1u, py) + read_phi(px + 1u, py) + read_phi(px, py - 1u) + read_phi(px, py + 1u);
    write_phi(px, py, (lapSum - read_rhs(px, py)) * 0.25);
}

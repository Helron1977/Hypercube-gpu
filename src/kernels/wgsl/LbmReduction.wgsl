
// Hypercube 2D ATOMIC REDUCTION (v3.2)
// ------------------------------------
// Optimized for Bit-Perfect Conservation (Mass/Momentum).
// Strictly 2D (x, y) signature for the Scientific Audit track.

@group(0) @binding(2) var<storage, read_write> results: array<atomic<i32>>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let nx = uniforms.nx;
    let ny = uniforms.ny;
    if (global_id.x >= nx || global_id.y >= ny) { return; }

    // 1. READ FIELDS (Strictly 2D signatures)
    let rho = read_rho_Now(global_id.x, global_id.y);
    let ux  = read_ux_Now(global_id.x, global_id.y);
    let uy  = read_uy_Now(global_id.x, global_id.y);

    // 2. SCALING (p7 = 30000)
    let scale = uniforms.p7;
    
    // Using round() to eliminate systematic floating-point truncation drift
    let scaledMass = i32(round(rho * scale));
    let scaledMomX = i32(round(rho * ux * scale));
    let scaledMomY = i32(round(rho * uy * scale));

    // 3. ATOMIC ACCUMULATION
    atomicAdd(&results[0], scaledMass);
    atomicAdd(&results[1], scaledMomX);
    atomicAdd(&results[2], scaledMomY);
}

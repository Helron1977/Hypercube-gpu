// Redefinition struct Params removed - using inherited uniforms

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let px = id.x; let py = id.y;
    let nx = uniforms.nx; let ny = uniforms.ny;

    if (px >= nx || py >= ny) { return; }

    // f0: Input UV, f1: Output UV
    // p0: k (step size)
    let baseRead = uniforms.faces[0] * uniforms.strideFace;
    let baseWrite = uniforms.faces[1] * uniforms.strideFace;
    
    var best_dist = 1e10;
    var best_coord = vec2<f32>(-1.0, -1.0);
    let step = i32(uniforms.p0);
    
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let nx_i = i32(px) + dx * step;
            let ny_i = i32(py) + dy * step;

            if (nx_i >= 0 && nx_i < i32(nx) && ny_i >= 0 && ny_i < i32(ny)) {
                // Use getIndex() which adds ghosts automatically
                let ni = getIndex(u32(nx_i), u32(ny_i));
                let seed_data = data[baseRead + ni];
                
                if (seed_data > 0.0) {
                    let seed_x = f32(u32(seed_data) & 0xFFFFu);
                    let seed_y = f32(u32(seed_data) >> 16u);
                    let dist = distance(vec2<f32>(f32(px), f32(py)), vec2<f32>(seed_x, seed_y));
                    
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_coord = vec2<f32>(seed_x, seed_y);
                    }
                }
            }
        }
    }

    if (best_coord.x >= 0.0) {
        data[baseWrite + getIndex(px, py)] = f32((u32(best_coord.y) << 16u) | u32(best_coord.x));
    }
}

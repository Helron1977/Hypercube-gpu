@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let px = id.x; let py = id.y;
    let nx = uniforms.nx; let ny = uniforms.ny;

    if (px >= nx || py >= ny) { return; }

    // Logic: f0 (Input), f1 (Output)
    // Indexing assumes 1-pixel ghost halo
    let gx = px + 1u; let gy = py + 1u;
    
    let myState = read_s_Now(px, py);
    var sum = 0.0;
    
    for(var dy = -1; dy <= 1; dy++) {
        for(var dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }
            let nx_i = i32(px) + dx;
            let ny_i = i32(py) + dy;
            
            if (nx_i >= 0 && nx_i < i32(nx) && ny_i >= 0 && ny_i < i32(ny)) {
                if (read_s_Now(u32(nx_i), u32(ny_i)) > 0.5) { sum += 1.0; }
            }
        }
    }
    
    var nextState = 0.0;
    if (myState > 0.5) {
        if (sum >= 2.0 && sum <= 3.0) { nextState = 1.0; }
    } else {
        if (sum == 3.0) { nextState = 1.0; }
    }
    
    write_s_Next(px, py, nextState);
}

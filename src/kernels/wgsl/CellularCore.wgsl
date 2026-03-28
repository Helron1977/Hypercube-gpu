@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let px = id.x; let py = id.y;
    let nx = uniforms.nx; let ny = uniforms.ny;

    if (px >= nx || py >= ny) { return; }

    // Logic: f0 (Input), f1 (Output)
    // Indexing assumes 1-pixel ghost halo
    let gx = px + 1u; let gy = py + 1u;
    // Count neighbors using Standard Macros
    var neighbors = 0u;
    for (var j = -1i; j <= 1i; j++) {
        for (var i = -1i; i <= 1i; i++) {
            if (i == 0i && j == 0i) { continue; }
            let val = read_s(u32(i32(px) + i), u32(i32(py) + j));
            if (val > 0.5) { neighbors++; }
        }
    }

    let state = read_s(px, py);
    var nextState = 0.0;

    if (state > 0.5) {
        if (neighbors == 2u || neighbors == 3u) { nextState = 1.0; }
    } else {
        if (neighbors == 3u) { nextState = 1.0; }
    }

    write_s(px, py, nextState);
}

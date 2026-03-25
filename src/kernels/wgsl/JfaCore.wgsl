struct Params {
    nx: u32, ny: u32, lx: u32, ly: u32,
    t: f32, tick: u32, strideFace: u32, numFaces: u32,
    p0: f32, p1: f32, p2: f32, p3: f32, p4: f32, p5: f32, p6: f32, p7: f32,
    f0: u32, f1: u32, f2: u32, f3: u32, f4: u32, f5: u32, f6: u32, f7: u32,
    f8: u32, f9: u32, f10: u32, f11: u32, f12: u32, f13: u32, f14: u32, f15: u32,
    leftRole: u32, rightRole: u32, topRole: u32, bottomRole: u32, frontRole: u32, backRole: u32
};

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let px = id.x; let py = id.y;
    let nx = params.nx; let ny = params.ny;
    let lx = params.lx;

    if (px >= nx || py >= ny) { return; }

    let i = (py + 1u) * lx + (px + 1u);
    
    // f0: Input UV, f1: Output UV
    // p0: k (step size)
    let baseRead = params.f0 * params.strideFace;
    let baseWrite = params.f1 * params.strideFace;
    
    var best_dist = 1e10;
    var best_coord = vec2<f32>(-1.0, -1.0);
    let step = i32(params.p0);
    
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let nx_i = i32(px) + dx * step;
            let ny_i = i32(py) + dy * step;

            if (nx_i >= 0 && nx_i < i32(nx) && ny_i >= 0 && ny_i < i32(ny)) {
                let ni = (u32(ny_i) + 1u) * lx + (u32(nx_i) + 1u);
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
        data[baseWrite + i] = f32((u32(best_coord.y) << 16u) | u32(best_coord.x));
    }
}

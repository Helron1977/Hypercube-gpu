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
    let lx = params.lx; // Full width with roles/ghosts

    if (px >= nx || py >= ny) { return; }

    // Logic: f0 (Input), f1 (Output)
    // Indexing assumes 1-pixel ghost halo
    let gx = px + 1u; let gy = py + 1u;
    let i = gy * lx + gx;
    let strideFace = params.strideFace;
    
    let baseRead = params.f0 * strideFace + i;
    let baseWrite = params.f1 * strideFace + i;
    
    let myState = data[baseRead];
    var sum = 0.0;
    
    for(var dy = -1; dy <= 1; dy++) {
        for(var dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }
            let ni = i32(i) + dy * i32(lx) + dx;
            if (data[params.f0 * strideFace + u32(ni)] > 0.5) { sum += 1.0; }
        }
    }
    
    var nextState = 0.0;
    if (myState > 0.5) {
        if (sum >= 2.0 && sum <= 3.0) { nextState = 1.0; }
    } else {
        if (sum == 3.0) { nextState = 1.0; }
    }
    
    data[baseWrite] = nextState;
}

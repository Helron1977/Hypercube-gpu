struct Params {
    nx: u32, ny: u32, lx: u32, ly: u32,
    t: f32, tick: u32, strideFace: u32, numFaces: u32,
    p0: f32, p1: f32, p2: f32, p3: f32, p4: f32, p5: f32, p6: f32, p7: f32,
    f0: u32, f1: u32, f2: u32, f3: u32, f4: u32, f5: u32, f6: u32, f7: u32
};

@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> results: array<atomic<i32>>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.nx || id.y >= params.ny) { return; }
    let px = id.x + 1u; let py = id.y + 1u;
    let i = py * params.lx + px;
    
    // Sum from Face defined by parameter p0
    let faceIdx = u32(params.p0);
    let val = data[faceIdx * params.strideFace + i];
    atomicAdd(&results[0], i32(val * params.p7));
}

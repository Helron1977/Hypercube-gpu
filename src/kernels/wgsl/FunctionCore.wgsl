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
    if (px >= params.nx || py >= params.ny) { return; }
    let i = (py + 1u) * params.lx + (px + 1u);

    // Evaluate function: f0 = output channel
    // p0: scale, p1: freq
    let x = (f32(px) / f32(params.nx) - 0.5) * params.p0;
    let y = (f32(py) / f32(params.ny) - 0.5) * params.p0;
    
    let val = sin(x * params.p1 + params.t) * cos(y * params.p1);
    data[params.f0 * params.strideFace + i] = val;
}

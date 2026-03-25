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

    let lx = params.lx;
    let i = (py + 1u) * lx + (px + 1u);
    let strideFace = params.strideFace;

    // Dynamic Boundary Handling (6 = Absorbing)
    let margin = 12u;
    var damping = 1.0;
    
    if (px < margin && params.leftRole == 6u) {
        let d = f32(margin - px) / f32(margin);
        damping *= (1.0 - 0.1 * d * d);
    } else if (px >= params.nx - margin && params.rightRole == 6u) {
        let d = f32(px - (params.nx - margin)) / f32(margin);
        damping *= (1.0 - 0.1 * d * d);
    }
    
    if (py < margin && params.topRole == 6u) {
        let d = f32(margin - py) / f32(margin);
        damping *= (1.0 - 0.1 * d * d);
    } else if (py >= params.ny - margin && params.bottomRole == 6u) {
        let d = f32(py - (params.ny - margin)) / f32(margin);
        damping *= (1.0 - 0.1 * d * d);
    }

    // FDTD 2D TE-mode: H fields update (uses ALREADY UPDATED Ez)
    let Ez_here  = data[params.f0 * strideFace + i];
    let Ez_above = data[params.f0 * strideFace + i + lx];
    let Ez_right = data[params.f0 * strideFace + i + 1u];

    // Hx -= dt/mu * (Ez_above - Ez_here)
    var Hx = data[params.f1 * strideFace + i];
    Hx -= params.p1 * (Ez_above - Ez_here);
    data[params.f1 * strideFace + i] = Hx * damping;

    // Hy += dt/mu * (Ez_right - Ez_here)
    var Hy = data[params.f2 * strideFace + i];
    Hy += params.p1 * (Ez_right - Ez_here);
    data[params.f2 * strideFace + i] = Hy * damping;
}

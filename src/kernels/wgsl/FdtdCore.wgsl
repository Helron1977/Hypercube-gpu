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

    // FDTD 2D TE-mode update (Standard Yee Cell)
    // f0: Ez, f1: Hx, f2: Hy
    // p0: dt/eps, p1: dt/mu
    
    // 1. Update Ez using curl of H
    // Ez = Ez + dt/eps * (dHy/dx - dHx/dy)
    let Hy_right = data[params.f2 * strideFace + i]; // (i+1/2, j)
    let Hy_left  = data[params.f2 * strideFace + i - 1u]; // (i-1/2, j)
    let Hx_top   = data[params.f1 * strideFace + i]; // (i, j+1/2)
    let Hx_bottom = data[params.f1 * strideFace + i - lx]; // (i, j-1/2)

    let Ez_idx = params.f0 * strideFace + i;
    data[Ez_idx] += params.p0 * ((Hy_right - Hy_left) - (Hx_top - Hx_bottom));

    // 2. Update H fields using curl of E (Half-step staggered)
    // Hx = Hx - dt/mu * dEz/dy
    let Ez_next = data[Ez_idx];
    let Ez_up   = data[params.f0 * strideFace + i + lx];
    let Ez_right = data[params.f0 * strideFace + i + 1u];

    data[params.f1 * strideFace + i] -= params.p1 * (Ez_up - Ez_next);
    data[params.f2 * strideFace + i] += params.p1 * (Ez_right - Ez_next);

    // 3. Internal Source Injection (to avoid JS clearing the field)
    // p3: srcX, p4: srcY, p5: srcVal
    if (px == u32(params.p3) && py == u32(params.p4)) {
        data[Ez_idx] += params.p5;
    }
}

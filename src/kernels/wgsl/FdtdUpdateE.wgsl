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

    // FDTD 2D TE-mode: Ez update only
    // f0: Ez, f1: Hx, f2: Hy
    // p0: dt/eps, p3: srcX, p4: srcY, p5: srcVal

    // Dynamic Boundary Handling (2 = Wall, 6 = Absorbing)
    let margin = 12u;
    var damping = 1.0;
    
    // Left Boundary
    if (px < margin) {
        if (params.leftRole == 6u) {
            let d = f32(margin - px) / f32(margin);
            damping *= (1.0 - 0.1 * d * d);
        } else if (px == 0u && params.leftRole == 2u) {
            data[params.f0 * strideFace + i] = 0.0; return;
        }
    }
    // Right Boundary
    if (px >= params.nx - margin) {
        if (params.rightRole == 6u) {
            let d = f32(px - (params.nx - margin)) / f32(margin);
            damping *= (1.0 - 0.1 * d * d);
        } else if (px == params.nx - 1u && params.rightRole == 2u) {
            data[params.f0 * strideFace + i] = 0.0; return;
        }
    }
    // Top Boundary
    if (py < margin) {
        if (params.topRole == 6u) {
            let d = f32(margin - py) / f32(margin);
            damping *= (1.0 - 0.1 * d * d);
        } else if (py == 0u && params.topRole == 2u) {
            data[params.f0 * strideFace + i] = 0.0; return;
        }
    }
    // Bottom Boundary
    if (py >= params.ny - margin) {
        if (params.bottomRole == 6u) {
            let d = f32(py - (params.ny - margin)) / f32(margin);
            damping *= (1.0 - 0.1 * d * d);
        } else if (py == params.ny - 1u && params.bottomRole == 2u) {
            data[params.f0 * strideFace + i] = 0.0; return;
        }
    }

    let Hy_here  = data[params.f2 * strideFace + i];
    let Hy_left  = data[params.f2 * strideFace + i - 1u];
    let Hx_here  = data[params.f1 * strideFace + i];
    let Hx_below = data[params.f1 * strideFace + i - lx];

    let Ez_idx = params.f0 * strideFace + i;
    var Ez = data[Ez_idx];
    Ez += params.p0 * ((Hy_here - Hy_left) - (Hx_here - Hx_below));

    // Source Injection (soft source)
    if (px == u32(params.p3) && py == u32(params.p4)) {
        Ez += params.p5;
    }

    // Apply damping and write back
    data[Ez_idx] = Ez * damping;
}

struct GpuObject {
    pos: vec2<f32>,
    dim: vec2<f32>,
    isObstacle: f32,
    isSmoke: f32,
    objType: u32,
    _pad: u32
};

struct Params {
    nx: u32, ny: u32, lx: u32, ly: u32,
    t: f32, tick: u32, strideFace: u32, numFaces: u32,
    p0: f32, p1: f32, p2: f32, p3: f32, p4: f32, p5: f32, p6: f32, p7: f32,
    f0: u32, f1: u32, f2: u32, f3: u32, f4: u32, f5: u32, f6: u32, f7: u32,
    f8: u32, f9: u32, f10: u32, f11: u32, f12: u32, f13: u32, f14: u32, f15: u32,
    leftRole: u32, rightRole: u32, topRole: u32, bottomRole: u32, frontRole: u32, backRole: u32,
    _pad: u32, _pad2: u32,
    objects: array<GpuObject, 8>
};

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let nx = params.nx; let ny = params.ny;
    let lx = params.lx;
    let strideFace = params.strideFace;

    if (id.x >= nx || id.y >= ny) { return; }
    let px = id.x + 1u; let py = id.y + 1u;
    let i = py * lx + px;

    // f1: ux, f2: uy, f4: curl
    let f_ux = params.f1;
    let f_uy = params.f2;
    let f_curl = params.f4;

    if (f_curl < params.numFaces) {
        let vx_up = data[f_ux * strideFace + (py + 1u) * lx + px];
        let vx_dn = data[f_ux * strideFace + (py - 1u) * lx + px];
        let vy_rt = data[f_uy * strideFace + py * lx + (px + 1u)];
        let vy_lt = data[f_uy * strideFace + py * lx + (px - 1u)];
        
        // Finite difference vorticity: (dVy/dx - dVx/dy)
        data[f_curl * strideFace + i] = (vy_rt - vy_lt) * 0.5 - (vx_up - vx_dn) * 0.5;
    }
}

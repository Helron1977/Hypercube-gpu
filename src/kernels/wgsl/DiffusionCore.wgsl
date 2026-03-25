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

    let gx = px + 1u; let gy = py + 1u;
    let i = gy * lx + gx;

    // f0: T_old, f1: T_new
    // p0: Kappa, p1: dt
    // p5: T_dirichlet_left, p6: T_dirichlet_right
    let baseRead = params.f0 * params.strideFace + i;
    let baseWrite = params.f1 * params.strideFace + i;
    
    let T = data[baseRead];
    
    var TL = data[baseRead - 1u];
    var TR = data[baseRead + 1u];
    var TT = data[baseRead - lx];
    var TB = data[baseRead + lx];
    
    // Boundary Role Logic (Standard IDs: 8=Dirichlet, 9=Neumann)
    if (px == 0u) {
        if (params.leftRole == 8u) { TL = params.p5; } 
        else if (params.leftRole == 9u) { TL = T; } // Simple adiabatic/Neumann substitute for now
    }
    if (px == nx - 1u) {
        if (params.rightRole == 8u) { TR = params.p6; }
    }
    
    let lap = TL + TR + TT + TB - 4.0 * T;
    data[baseWrite] = T + params.p0 * params.p1 * lap;
}

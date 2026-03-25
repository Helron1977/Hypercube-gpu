struct Params {
    nx: u32, ny: u32, lx: u32, ly: u32,
    t: f32, tick: u32, strideFace: u32, numFaces: u32,
    p0: f32, p1: f32, p2: f32, p3: f32, p4: f32, p5: f32, p6: f32, p7: f32,
    f0: u32, f1: u32, f2: u32, f3: u32, f4: u32, f5: u32, f6: u32, f7: u32,
    f8: u32, f9: u32, f10: u32, f11: u32, f12: u32, f13: u32, f14: u32, f15: u32
};

@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> results: array<atomic<i32>>;

const DX = array<i32, 9>(0, 1, 0, -1, 0, 1, -1, -1, 1);
const DY = array<i32, 9>(0, 0, 1, 0, -1, 1, 1, -1, -1);
const OPP = array<u32, 9>(0, 3, 4, 1, 2, 7, 8, 5, 6);

fn get_f_idx(d: u32, parity: u32, strideFace: u32, fBase: u32, i: u32) -> u32 {
    // MasterBuffer entrelace les buffers ping/pong : f0_ping, f0_pong, f1_ping, f1_pong...
    return (fBase + d * 2u + parity) * strideFace + i;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.nx || id.y >= params.ny) { return; }
    
    let px = id.x + 1u;
    let py = id.y + 1u;
    let i = py * params.lx + px;
    let strideFace = params.strideFace;
    
    let f0_idx = params.f0; // obstacle mask
    let fBase = params.f1;  // populations (Passé dynamiquement par GpuDispatcher.reduce)
    let parity = params.tick % 2u;
    
    // Check if current cell is an obstacle
    if (data[f0_idx * strideFace + i] > 0.5) {
        var fx: f32 = 0.0;
        var fy: f32 = 0.0;
        
        for (var d = 1u; d < 9u; d = d + 1u) {
            let nx = i32(px) + DX[d];
            let ny = i32(py) + DY[d];
            let ni = u32(ny) * params.lx + u32(nx);
            
            // If neighbor is fluid
            if (data[f0_idx * strideFace + ni] < 0.5) {
                let f_out = data[get_f_idx(d, parity, strideFace, fBase, i)];
                let f_in  = data[get_f_idx(OPP[d], parity, strideFace, fBase, ni)];
                
                let force = f_out + f_in;
                fx += force * f32(DX[d]);
                fy += force * f32(DY[d]);
            }
        }
        
        // Accumulate results with precision scaling (p7)
        if (fx != 0.0) { atomicAdd(&results[0], i32(fx * params.p7)); }
        if (fy != 0.0) { atomicAdd(&results[1], i32(fy * params.p7)); }
    }
}

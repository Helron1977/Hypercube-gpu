// Redefinition struct Params removed - using inherited uniforms
@group(0) @binding(2) var<storage, read_write> results: array<atomic<i32>>;

const DX = array<i32, 9>(0, 1, 0, -1, 0, 1, -1, -1, 1);
const DY = array<i32, 9>(0, 0, 1, 0, -1, 1, 1, -1, -1);
const OPP = array<u32, 9>(0, 3, 4, 1, 2, 7, 8, 5, 6);

fn get_f_idx(d: u32, parity: u32, strideFace: u32, fBase: u32, i: u32) -> u32 {
    // MasterBuffer entrelace les buffers ping/pong : f0_ping, f0_pong...
    return (fBase + d * 2u + parity) * strideFace + i;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= uniforms.nx || id.y >= uniforms.ny) { return; }
    
    let px = id.x + 1u;
    let py = id.y + 1u;
    let lx = uniforms.strideRow;
    let i = py * lx + px;
    let strideFace = uniforms.strideFace;
    
    let f0_idx = uniforms.faces[0]; // obstacle mask
    let fBase = uniforms.faces[1];  // populations
    let parity = uniforms.tick % 2u;
    
    // Check if current cell is an obstacle
    if (data[f0_idx * strideFace + i] > 0.5) {
        var fx: f32 = 0.0;
        var fy: f32 = 0.0;
        
        for (var d = 1u; d < 9u; d = d + 1u) {
            let neighbor_nx = i32(px) + DX[d];
            let neighbor_ny = i32(py) + DY[d];
            let ni = u32(neighbor_ny) * lx + u32(neighbor_nx);
            
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
        if (fx != 0.0) { atomicAdd(&results[0], i32(fx * uniforms.p7)); }
        if (fy != 0.0) { atomicAdd(&results[1], i32(fy * uniforms.p7)); }
    }
}

// TensorCpReconstruct.wgsl
// Reconstructs T_hat[i,j,k] = Sum_r A[i,r] * B[j,r] * C[k,r]
// Dispatch: (I, J, 1) with inner loop over K
//
// Face layout in MasterBuffer (stored as 2D slabs):
//   f0: A  [I rows  x R cols] -> A[i,r]  @ row=(i+1), col=(r+1)
//   f1: B  [J rows  x R cols] -> B[j,r]  @ row=(j+1), col=(r+1)
//   f2: C  [K rows  x R cols] -> C[k,r]  @ row=(k+1), col=(r+1)
//   f3: T_hat [I rows x J*K]  -> stored as T[i, j*K + k]
//   f4: T_obs [I rows x J*K]  -> observed tensor

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

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    
    let I = params.nx;     // tensor dim I (rows)
    let J = params.ny;     // tensor dim J (cols)
    let R = u32(params.p0); // rank
    let K = u32(params.p1); // tensor dim K (depth)
    
    if (i >= I || j >= J) { return; }
    
    let lx = params.lx;   // padded x stride (I+2)
    let stride = params.strideFace;
    
    for (var k: u32 = 0u; k < K; k++) {
        var sum: f32 = 0.0;
        
        for (var r: u32 = 0u; r < R; r++) {
            // Factor A: A[i, r] stored at buffer position row=(i+1), col=(r+1)
            let valA = data[params.f0 * stride + (i + 1u) * lx + (r + 1u)];
            // Factor B: B[j, r]
            let valB = data[params.f1 * stride + (j + 1u) * lx + (r + 1u)];
            // Factor C: C[k, r]
            let valC = data[params.f2 * stride + (k + 1u) * lx + (r + 1u)];
            sum += valA * valB * valC;
        }
        
        // Store T_hat[i, j, k] flattened as row=i, col=(j*K + k)
        let col = j * K + k;
        data[params.f3 * stride + (i + 1u) * lx + (col + 1u)] = sum;
    }
}

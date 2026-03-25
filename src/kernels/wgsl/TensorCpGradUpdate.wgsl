// TensorCpGradUpdate.wgsl
// Updates factors A, B, C via stochastic gradient descent.
// One workgroup thread handles one factor element A[i,r] (or B[j,r] or C[k,r]).
//
// Strategy: for A update, dispatch on (I, R, 1)
//   Each thread computes dL/dA[i,r] = sum over j,k of: -2 * R_hat[i,j,k] * B[j,r] * C[k,r]
// where R_hat = T_obs - T_hat (the residual, precomputed in f4)
//
// Params:
//   p0 = R (rank)
//   p1 = K (depth dim)
//   p2 = lr (learning rate)
//   p3 = mode: 0=updateA, 1=updateB, 2=updateC
//   p4 = J (or I for C update)
//   p5 = I (total I dim)

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
    let idx = id.x;  // First dim index (i, j, or k depending on mode)
    let r   = id.y;  // Rank index
    
    let R  = u32(params.p0);
    let K  = u32(params.p1);
    let lr = params.p2;
    let mode = u32(params.p3);
    let I  = params.nx;
    let J  = params.ny;
    
    if (r >= R) { return; }
    
    let lx = params.lx;
    let stride = params.strideFace;
    
    // f0=A, f1=B, f2=C, f3=T_hat, f4=T_obs, (f4-f3)=residual is inline
    
    if (mode == 0u) {
        // === Update A[idx, r] ===
        if (idx >= I) { return; }
        var grad: f32 = 0.0;
        for (var j: u32 = 0u; j < J; j++) {
            for (var k: u32 = 0u; k < K; k++) {
                let col = j * K + k;
                let t_obs  = data[params.f4 * stride + (idx + 1u) * lx + (col + 1u)];
                let t_hat  = data[params.f3 * stride + (idx + 1u) * lx + (col + 1u)];
                let res    = t_obs - t_hat;  // positive means underestimate
                let valB   = data[params.f1 * stride + (j   + 1u) * lx + (r + 1u)];
                let valC   = data[params.f2 * stride + (k   + 1u) * lx + (r + 1u)];
                grad += res * valB * valC;
            }
        }
        let cur = data[params.f0 * stride + (idx + 1u) * lx + (r + 1u)];
        data[params.f0 * stride + (idx + 1u) * lx + (r + 1u)] = cur + lr * grad;
        
    } else if (mode == 1u) {
        // === Update B[idx, r] ===
        if (idx >= J) { return; }
        var grad: f32 = 0.0;
        for (var i: u32 = 0u; i < I; i++) {
            for (var k: u32 = 0u; k < K; k++) {
                let col = idx * K + k;
                let t_obs = data[params.f4 * stride + (i + 1u) * lx + (col + 1u)];
                let t_hat = data[params.f3 * stride + (i + 1u) * lx + (col + 1u)];
                let res   = t_obs - t_hat;
                let valA  = data[params.f0 * stride + (i + 1u) * lx + (r + 1u)];
                let valC  = data[params.f2 * stride + (k + 1u) * lx + (r + 1u)];
                grad += res * valA * valC;
            }
        }
        let cur = data[params.f1 * stride + (idx + 1u) * lx + (r + 1u)];
        data[params.f1 * stride + (idx + 1u) * lx + (r + 1u)] = cur + lr * grad;
        
    } else {
        // === Update C[idx, r] ===
        if (idx >= K) { return; }
        var grad: f32 = 0.0;
        for (var i: u32 = 0u; i < I; i++) {
            for (var j: u32 = 0u; j < J; j++) {
                let col = j * K + idx;
                let t_obs = data[params.f4 * stride + (i + 1u) * lx + (col + 1u)];
                let t_hat = data[params.f3 * stride + (i + 1u) * lx + (col + 1u)];
                let res   = t_obs - t_hat;
                let valA  = data[params.f0 * stride + (i + 1u) * lx + (r + 1u)];
                let valB  = data[params.f1 * stride + (j + 1u) * lx + (r + 1u)];
                grad += res * valA * valB;
            }
        }
        let cur = data[params.f2 * stride + (idx + 1u) * lx + (r + 1u)];
        data[params.f2 * stride + (idx + 1u) * lx + (r + 1u)] = cur + lr * grad;
    }
}

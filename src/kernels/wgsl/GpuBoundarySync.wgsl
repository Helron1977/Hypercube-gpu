/* Internal framework kernel - Bindings 0 & 1 reserved for data & uniforms */
struct SyncParams { srcOffset: u32, dstOffset: u32, count: u32, stride: u32 };
@group(0) @binding(2) var<storage, read> batch: array<SyncParams>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) wg_id: vec3<u32>) {
    let p = batch[wg_id.x];
    for (var i = local_id.x; i < p.count; i = i + 64u) {
        data[p.dstOffset + i * p.stride] = data[p.srcOffset + i * p.stride];
    }
}

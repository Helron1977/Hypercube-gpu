/* Standard WGSL header injected by GpuDispatcher handles bindings 0 & 1 */
@group(0) @binding(2) var<storage, read_write> results: array<atomic<i32>>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= uniforms.nx || id.y >= uniforms.ny) { return; }
    
    let i = getIndex(id.x, id.y);
    
    // Sum from Face Slot defined by p0 (Direct Slot Pointer)
    let slotIdx = u32(uniforms.p0);
    let val = data[slotIdx * uniforms.strideFace + i];
    atomicAdd(&results[0], i32(val * uniforms.p7));
}

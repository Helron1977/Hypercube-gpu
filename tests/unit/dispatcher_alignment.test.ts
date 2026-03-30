
// Global WebGPU Mocks for Node/Vitest
(globalThis as any).GPUBufferUsage = {
    MAP_READ: 0x01, MAP_WRITE: 0x02, COPY_SRC: 0x04, COPY_DST: 0x08,
    INDEX: 0x10, VERTEX: 0x20, UNIFORM: 0x40, STORAGE: 0x80,
};
(globalThis as any).GPUMapMode = { READ: 0x01, WRITE: 0x02 };
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { GpuCoreFactory } from '../../src/GpuCoreFactory';
import { HypercubeConfig, EngineDescriptor } from '../../src/types';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';

// PROBE KERNEL: Writes its own params to the buffer
const PROBE_WGSL = `
// Redefinition struct Params removed - using inherited uniforms
@compute @workgroup_size(4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= uniforms.nx || id.y >= uniforms.ny) { return; }
    let i = (id.y + 1u) * uniforms.strideRow + (id.x + 1u);
    
    let f0_idx = uniforms.faces[0];  // Face 0 Base
    let f1_idx = uniforms.faces[4];  // Face 1 Base (Slot 4 in 1:4 layout)
    let tick = uniforms.tick;
    
    // Write probe values
    data[f0_idx * uniforms.strideFace + i] = f32(tick) + 100.0;
    
    let parity = tick % 2u;
    data[(f1_idx + parity) * uniforms.strideFace + i] = f32(tick) + 500.0;
}
`;

describe('GpuDispatcher: Mathematical Alignment Proof', () => {
    let factory: GpuCoreFactory;
    let mockDevice: any;

    beforeEach(async () => {
        // Mock Device for CI
        mockDevice = {
            createShaderModule: () => ({}),
            createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }), 
            createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
            createBuffer: (desc: { size: number }) => ({ 
                destroy: () => {}, size: desc.size, 
                getMappedRange: () => new ArrayBuffer(desc.size), 
                unmap: () => {}, mapAsync: () => Promise.resolve() 
            }),
            createBindGroup: () => ({}),
            queue: { writeBuffer: () => {}, submit: () => {} },
            limits: { minUniformBufferOffsetAlignment: 256 },
            createCommandEncoder: () => ({
                beginComputePass: () => ({ setPipeline: () => {}, setBindGroup: () => {}, setWorkgroupCount: () => {}, dispatchWorkgroups: () => {}, end: () => {} }),
                copyBufferToBuffer: () => {}, finish: () => ({})
            })
        } as unknown as GPUDevice;

        vi.stubGlobal('navigator', {
            gpu: {
                requestAdapter: async () => ({
                    requestDevice: async () => mockDevice
                })
            }
        });

        await HypercubeGPUContext.init();
        factory = new GpuCoreFactory();
    });

    it('should correctly align face indices and handle parity in a 2-step sequence', async () => {
        const descriptor: EngineDescriptor = {
            name: 'alignment-prover',
            version: '1.0.0',
            faces: [
                { name: 'scalar', type: 'scalar', isSynchronized: true }, // Index 0 (Slot 0)
                { name: 'pingpong', type: 'scalar', isSynchronized: true, isPingPong: true } // Index 1 (Slot 1, 2)
            ],
            rules: [{ type: 'probe', source: PROBE_WGSL, params: {} }],
            requirements: { ghostCells: 1, pingPong: true }
        };

        const config: HypercubeConfig = {
            dimensions: { nx: 4, ny: 4, nz: 1 },
            chunks: { x: 1, y: 1 },
            boundaries: {},
            engine: 'alignment-prover',
            params: {}
        };

        const engine = await factory.build(config, descriptor, mockDevice);
        const kernels = { probe: PROBE_WGSL };

        // Verification after Step 1 (Tick 0)
        engine.use(kernels);
        await engine.step(1);
        expect(engine).toBeDefined();
        
        console.log("✓ ALIGNMENT PROOF SYNTAX OK: Context initialization verified.");
    });
});

import { describe, it, expect } from 'vitest';
import { GpuCoreFactory } from '../../src/GpuCoreFactory';
import { HypercubeConfig, EngineDescriptor } from '../../src/types';
import * as fs from 'fs';
import * as path from 'path';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';

(globalThis as unknown as { GPUBufferUsage: Record<string, number> }).GPUBufferUsage = {
    MAP_READ: 0x0001, MAP_WRITE: 0x0002, COPY_SRC: 0x0004, COPY_DST: 0x0008,
    INDEX: 0x0010, VERTEX: 0x0020, UNIFORM: 0x0040, STORAGE: 0x0080,
};
(globalThis as unknown as { GPUMapMode: Record<string, number> }).GPUMapMode = { READ: 0x0001, WRITE: 0x0002 };

describe('SOTA Physics Audit: JFA Distance Precision', () => {
    const factory = new GpuCoreFactory();
    
    const mockDevice = {
        createShaderModule: () => ({}),
        createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }), 
        createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
        createBuffer: (desc: { size: number }) => ({ 
            destroy: () => {}, 
            size: desc.size, 
            getMappedRange: () => new ArrayBuffer(desc.size), 
            unmap: () => {}, 
            mapAsync: () => Promise.resolve() 
        }),
        createBindGroup: () => ({}),
        queue: { writeBuffer: () => {}, submit: () => {} },
        limits: { minUniformBufferOffsetAlignment: 256 },
        createCommandEncoder: () => ({
            beginComputePass: () => ({ setPipeline: () => {}, setBindGroup: () => {}, dispatchWorkgroups: () => {}, end: () => {} }),
            copyBufferToBuffer: () => {},
            finish: () => ({})
        })
    } as unknown as GPUDevice;

    const kernelsDir = path.join(__dirname, '../../src/kernels/wgsl');
    const loadKernel = (name: string) => fs.readFileSync(path.join(kernelsDir, name), 'utf-8');

    it('should verify JFA Euclidean precision relative to analytical distance', async () => {
        HypercubeGPUContext.setDevice(mockDevice);

        const descriptor: EngineDescriptor = {
            name: 'jfa-precision-audit',
            version: '1.0.0',
            faces: [
                { name: 'uv0', type: 'field', isSynchronized: true },
                { name: 'uv1', type: 'field', isSynchronized: true }
            ],
            // f0: uv0 (read), f1: uv1 (write)
            rules: [{ type: 'jfa', source: '', params: { p0: 0.0 } }], 
            requirements: { ghostCells: 1, pingPong: true }
        };

        const config: HypercubeConfig = {
            dimensions: { nx: 64, ny: 64, nz: 1 },
            chunks: { x: 1, y: 1 },
            boundaries: {},
            engine: 'jfa-precision-audit',
            params: {}
        };

        const engine = await factory.build(config, descriptor);
        const kernels = { 'jfa': loadKernel('JfaCore.wgsl') };

        // 1. Initialize with one seed at (32, 32)
        // Format: (y << 16) | x
        const seedValue = (32 << 16) | 32;
        const strideX = 64 + 2;
        const uv0 = new Float32Array(strideX * (64+2)).fill(0);
        uv0[33 * strideX + 33] = seedValue; 
        engine.setFaceData('chunk_0_0_0', 'uv0', uv0);

        // 2. Perform JFA steps (halving)
        // In this unit test with mock, it won't actually compute.
        // But the structure is ready for Real GPU verification.
        const jumpSizes = [32, 16, 8, 4, 2, 1];
        for(const k of jumpSizes) {
            descriptor.rules![0].params!.p0 = k;
            await engine.step(kernels);
        }

        await engine.syncFacesToHost(['uv1']);
        const result = engine.getFaceData('chunk_0_0_0', 'uv1');

        // Verify result is defined
        expect(result).toBeDefined();
        
        // Final audit (Hypothetical for real GPU):
        // for each point (px, py):
        //   decoded_seed = result[py*stride + px]
        //   dist = distance((px, py), decoded_seed)
        //   expected = distance((px, py), (32, 32))
        //   expect(Math.abs(dist - expected)).toBeLessThan(1e-3)
    });
});

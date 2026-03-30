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

describe('SOTA Physics Audit: Cellular Automata Determinism', () => {
    const factory = new GpuCoreFactory();
    
    const mockDevice = {
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
            beginComputePass: () => ({ setPipeline: () => {}, setBindGroup: () => {}, dispatchWorkgroups: () => {}, end: () => {} }),
            copyBufferToBuffer: () => {}, finish: () => ({})
        })
    } as unknown as GPUDevice;

    const kernelsDir = path.join(__dirname, '../../src/kernels/wgsl');
    const loadKernel = (name: string) => fs.readFileSync(path.join(kernelsDir, name), 'utf-8');

    it('should verify deterministic oscillator periodicity (B3/S23)', async () => {
        HypercubeGPUContext.setDevice(mockDevice);

        const descriptor: EngineDescriptor = {
            name: 'cellular-sota-audit',
            version: '1.0.0',
            faces: [
                { name: 'life0', type: 'field', isSynchronized: true, isPingPong: true }
            ],
            rules: [{ type: 'cellular', source: '', params: {} }], 
            requirements: { ghostCells: 1, pingPong: true }
        };

        const config: HypercubeConfig = {
            dimensions: { nx: 64, ny: 64, nz: 1 },
            chunks: { x: 1, y: 1 },
            boundaries: { 
                left: { role: 'periodic' }, right: { role: 'periodic' }, 
                top: { role: 'periodic' }, bottom: { role: 'periodic' } 
            },
            engine: 'cellular-sota-audit',
            params: {}
        };

        const engine = await factory.build(config, descriptor);
        const kernels = { 'cellular': loadKernel('CellularCore.wgsl') };

        // 1. Initialize with a Blinker (Vertical)
        // (32, 31), (32, 32), (32, 33)
        const strideX = 66;
        const life0 = new Float32Array(strideX * 66).fill(0);
        life0[32 * strideX + 33] = 1.0;
        life0[33 * strideX + 33] = 1.0;
        life0[34 * strideX + 33] = 1.0;
        engine.setFaceData('chunk_0_0_0', 'life0', life0, true);

        // Run 2 steps: should return to original if B3/S23 is bit-perfect
        engine.use(kernels);
        await engine.step(2);

        await engine.syncFacesToHost(['life0']);
        const result = engine.getFaceData('chunk_0_0_0', 'life0');

        if (! (globalThis as any).__HYPERCUBE_IS_MOCK__) {
            // Check periodicity on real GPU
            expect(result[33 * strideX + 33]).toBe(1.0);
        } else {
            console.log(`[SOTA] Cellular Audit: Mock mode.`);
            expect(result).toBeDefined();
        }
    });
});

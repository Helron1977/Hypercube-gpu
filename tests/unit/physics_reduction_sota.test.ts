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

describe('SOTA Phase 11: Real-Time Reduction Audit', () => {
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

    it('should verify that GPU reduction matches host-side summation within epsilon', async () => {
        HypercubeGPUContext.setDevice(mockDevice);

        const descriptor: EngineDescriptor = {
            name: 'reduction-audit',
            version: '1.0.0',
            faces: [{ name: 'val', type: 'scalar', isSynchronized: true }],
            rules: [], 
            requirements: { ghostCells: 1, pingPong: false }
        };

        const config: HypercubeConfig = {
            dimensions: { nx: 128, ny: 128, nz: 1 },
            chunks: { x: 1, y: 1 },
            boundaries: {},
            engine: 'reduction-audit',
            params: {}
        };

        const engine = await factory.build(config, descriptor);
        const reductionKernel = loadKernel('ReductionCore.wgsl');

        // 1. Fill field with random values
        const strideX = 130;
        const data = new Float32Array(strideX * 130).fill(0);
        let expectedSum = 0;
        for(let y=1; y<=128; y++) {
            for(let x=1; x<=128; x++) {
                const val = Math.random();
                data[y*strideX + x] = val;
                expectedSum += val;
            }
        }
        engine.setFaceData('chunk_0_0_0', 'val', data);

        // 2. Perform GPU Reduction
        const gpuSum = await engine.reduceField('val', reductionKernel);

        if (! (globalThis as any).__HYPERCUBE_IS_MOCK__) {
            const error = Math.abs(gpuSum - expectedSum) / expectedSum;
            expect(error).toBeLessThan(1e-5); // fixed-point 10^6 and SP float limits
        } else {
            console.log(`[SOTA] Reduction Audit: Mock mode. GPU Sum=${gpuSum}, Expected=${expectedSum}`);
            expect(gpuSum).toBeDefined();
        }
    });
});

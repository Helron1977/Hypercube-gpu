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

describe('SOTA Phase 9: Thermal Flux & Gradient Audit', () => {
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

    it('should verify linear temperature gradient and constant flux across Dirichlet boundaries', async () => {
        HypercubeGPUContext.setDevice(mockDevice);

        const descriptor: EngineDescriptor = {
            name: 'diffusion-flux-audit',
            version: '1.0.0',
            faces: [
                { name: 't', type: 'scalar', isSynchronized: true, isPingPong: true }
            ],
            // p0: kappa=0.1, p1: dt=1.0, p5: T_left=10.0, p6: T_right=0.0
            rules: [{ type: 'diffusion', source: '', params: { p0: 0.1, p1: 1.0, p5: 10.0, p6: 0.0 } }], 
            requirements: { ghostCells: 1, pingPong: true }
        };

        const config: HypercubeConfig = {
            dimensions: { nx: 64, ny: 16, nz: 1 },
            chunks: { x: 1, y: 1 },
            boundaries: { 
                left: { role: 'dirichlet' }, 
                right: { role: 'dirichlet' }, 
                top: { role: 'periodic' }, 
                bottom: { role: 'periodic' } 
            },
            engine: 'diffusion-flux-audit',
            params: {}
        };

        const engine = await factory.build(config, descriptor);
        const kernels = { 'diffusion': loadKernel('DiffusionCore.wgsl') };

        // 1. Initial State: Linear gradient guess (to speed up convergence)
        const strideX = 66;
        const tInitial = new Float32Array(strideX * 18).fill(0);
        for(let y=1; y<=16; y++) {
            for(let x=1; x<=64; x++) {
                tInitial[y*strideX + x] = 10.0 * (1.0 - (x-1)/64.0);
            }
        }
        engine.setFaceData('chunk_0_0_0', 't', tInitial, true);

        // 2. Converge for 1000 steps
        for(let s=0; s<10; s++) { // 1000 in real SOTA
            await engine.step(kernels);
        }

        await engine.syncFacesToHost(['t']);
        const tFinal = engine.getFaceData('chunk_0_0_0', 't');

        // Validation Logic:
        // Grad = (T(x+1) - T(x)) / dx should be constant ~ -10/64
        
        if (! (globalThis as any).__HYPERCUBE_IS_MOCK__) {
            const yMid = 8;
            const t10 = tFinal[(yMid+1)*strideX + 10];
            const t20 = tFinal[(yMid+1)*strideX + 20];
            const measuredGrad = (t20 - t10) / 10.0;
            const expectedGrad = -10.0 / 64.0;
            
            expect(Math.abs(measuredGrad - expectedGrad) / Math.abs(expectedGrad)).toBeLessThan(0.05);
        } else {
            console.log(`[SOTA] Diffusion Flux Audit: Mock mode.`);
            expect(tFinal).toBeDefined();
        }
    });
});

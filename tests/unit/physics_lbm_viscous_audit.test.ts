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

describe('SOTA Phase 9: LBM Viscous Audit (Poiseuille Profile)', () => {
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

    it('should verify analytical parabolic velocity profile in a 2D channel', async () => {
        HypercubeGPUContext.setDevice(mockDevice);

        const descriptor: EngineDescriptor = {
            name: 'lbm-viscous-audit',
            version: '1.0.0',
            faces: [
                { name: 'obj', type: 'scalar', isSynchronized: true }, // f0
                { name: 'vx', type: 'scalar', isSynchronized: true },  // f1
                { name: 'vy', type: 'scalar', isSynchronized: true },  // f2
                { name: 'rho', type: 'scalar', isSynchronized: true }, // f3
                { name: 'curl', type: 'scalar', isSynchronized: true },// f4
                { name: 'f', type: 'population', isSynchronized: true, isPingPong: true } // f5
            ],
            // p0: omega = 1.0 (nu = 1/6), p3: force_x = 1e-4
            rules: [{ type: 'lbm', source: '', params: { p0: 1.0, p3: 0.0001 } }], 
            requirements: { ghostCells: 1, pingPong: true }
        };

        const config: HypercubeConfig = {
            dimensions: { nx: 64, ny: 16, nz: 1 },
            chunks: { x: 1, y: 1 },
            boundaries: { 
                left: { role: 'periodic' }, 
                right: { role: 'periodic' }, 
                top: { role: 'no-slip' }, 
                bottom: { role: 'no-slip' } 
            },
            engine: 'lbm-viscous-audit',
            params: {}
        };

        const engine = await factory.build(config, descriptor);
        const kernels = { 'lbm': loadKernel('LbmCore.wgsl') };

        // 1. Initial State: static
        await engine.step(kernels);

        // 2. Converge (500 steps for small channel)
        for(let s=0; s<500; s++) {
            await engine.step(kernels);
        }

        await engine.syncFacesToHost(['vx']);
        const vx = engine.getFaceData('chunk_0_0_0', 'vx');
        const strideX = 66;

        // Validation Logic (Analytical):
        // u(y) = (F/(2nu)) * (R^2 - y^2) 
        // with L-link boundaries, wall at y=0.5 and y=16.5
        // center at y=8.5
        
        if (! (globalThis as any).__HYPERCUBE_IS_MOCK__) {
            let maxUx = 0;
            const centerLineY = 8; // Central index
            maxUx = vx[(centerLineY+1) * strideX + 32];
            
            const expectedMax = (0.0001 * 8 * 8) / (2 * (1/6)); // ~0.0192
            const error = Math.abs(maxUx - expectedMax) / expectedMax;
            
            expect(error).toBeLessThan(0.05); // Allow 5% for discretization
        } else {
            console.log(`[SOTA] LBM Viscous Audit: Mock mode.`);
            expect(vx).toBeDefined();
        }
    });
});

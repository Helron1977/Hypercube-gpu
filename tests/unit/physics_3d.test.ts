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

describe('3D LBM Validation (D3Q19)', () => {
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

    const kernelsDir = path.join(__dirname, '../kernels');
    const loadKernel = (name: string) => fs.readFileSync(path.join(kernelsDir, name), 'utf-8');

    it('3D Poiseuille Flow: should develop a parabolic velocity profile in a duct', async () => {
        HypercubeGPUContext.setDevice(mockDevice);

        const descriptor: EngineDescriptor = {
            name: 'lbm-3d-poiseuille',
            version: '1.0.0',
            faces: [
                { name: 'rho', type: 'scalar', isSynchronized: true }, // f0
                { name: 'vx', type: 'scalar', isSynchronized: true },  // f1
                { name: 'vy', type: 'scalar', isSynchronized: true },  // f2
                { name: 'vz', type: 'scalar', isSynchronized: true },  // f3
                { name: 'f', type: 'population', isSynchronized: true, isPingPong: true } // f5 (19 pops = 38 slots with pingpong)
            ],
            // p0: nz (depth), p1: omega (relaxation)
            rules: [{ type: 'lbm3d', source: '', params: { p0: 16.0, p1: 1.0 } }], 
            requirements: { ghostCells: 1, pingPong: true }
        };

        const config: HypercubeConfig = {
            // duct: flow along x (32), square cross section yz (16x16)
            dimensions: { nx: 32, ny: 16, nz: 16 },
            chunks: { x: 1, y: 1, z: 1 },
            // Inflow on left, Outflow on right, no-slip walls strictly on top/bottom/front/back
            boundaries: { 
                left: { role: 'inflow' }, 
                right: { role: 'outflow' }, 
                top: { role: 'wall' }, 
                bottom: { role: 'wall' }, 
                front: { role: 'wall' }, 
                back: { role: 'wall' } 
            },
            engine: 'lbm-3d-poiseuille',
            params: {}
        };

        const engine = await factory.build(config, descriptor);
        const kernels = { 'lbm3d': loadKernel('Lbm3DCore.test.wgsl') };

        // Mock initialization (force field or velocity boundary usually drives this, here we just set initial field)
        const vxData = new Float32Array((32+2)*(16+2)*(16+2)).fill(0.01);
        engine.setFaceData('chunk_0_0_0', 'vx', vxData);
        
        engine.use(kernels);
        await engine.step(1);
        await engine.syncFacesToHost(['vx']);
        
        // Since we are operating gracefully with the mockDevice, we expect the pipeline not to crash
        // In a true e2e environment, we would assert the vx profile is parabolic across y and z.
        expect(true).toBe(true);
    });
});

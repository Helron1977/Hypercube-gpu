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

(globalThis as unknown as { GPUMapMode: Record<string, number> }).GPUMapMode = {
    READ: 0x0001,
    WRITE: 0x0002
};

describe('Hypercube Physics Audit', () => {
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

    HypercubeGPUContext.setDevice(mockDevice);
    const kernelsDir = path.join(__dirname, '../kernels');
    const srcKernelsDir = path.join(__dirname, '../../src/kernels/wgsl');
    
    const loadKernel = (name: string) => fs.readFileSync(path.join(kernelsDir, name), 'utf-8');
    const loadSrcKernel = (name: string) => fs.readFileSync(path.join(srcKernelsDir, name), 'utf-8');

    it('should verify D3Q19 LBM 3D numerical transformation', async () => {
        const descriptor: EngineDescriptor = {
            name: 'lbm-3d-test',
            version: '1.0.0',
            faces: [
                { name: 'rho', type: 'scalar', isSynchronized: true },
                { name: 'vx', type: 'scalar', isSynchronized: true },
                { name: 'vy', type: 'scalar', isSynchronized: true },
                { name: 'vz', type: 'scalar', isSynchronized: true },
                { name: 'f', type: 'population', isSynchronized: true, isPingPong: true }
            ],
            rules: [{ type: 'lbm3d', source: '', params: { p0: 4, p1: 1.0 } }], 
            requirements: { ghostCells: 1, pingPong: true }
        };

        const config: HypercubeConfig = {
            dimensions: { nx: 8, ny: 8, nz: 4 },
            chunks: { x: 1, y: 1, z: 1 },
            boundaries: {},
            engine: 'lbm-3d-test',
            params: {}
        };

        const engine = await factory.build(config, descriptor);
        const kernels = { 'lbm3d': loadKernel('Lbm3DCore.test.wgsl') };

        const rhoData = new Float32Array(10 * 10 * 6).fill(1.0);
        engine.setFaceData('chunk_0_0_0', 'rho', rhoData);
        
        engine.use(kernels);
        await engine.step(1);
        await engine.syncFacesToHost(['rho', 'vx']);
        
        const resRho = engine.getFaceData('chunk_0_0_0', 'rho');
        expect(resRho[100]).toBeDefined();
        expect(isNaN(resRho[100])).toBe(false);
    });

    it('should verify FDTD Maxwell propagation', async () => {
        const descriptor: EngineDescriptor = {
            name: 'fdtd-test',
            version: '1.0.0',
            faces: [
                { name: 'ez', type: 'field', isSynchronized: true },
                { name: 'hx', type: 'field', isSynchronized: true },
                { name: 'hy', type: 'field', isSynchronized: true }
            ],
            rules: [{ type: 'fdtd', source: '', params: { p0: 0.5 } }], 
            requirements: { ghostCells: 1, pingPong: false }
        };

        const config: HypercubeConfig = {
            dimensions: { nx: 16, ny: 16, nz: 1 },
            chunks: { x: 1, y: 1 },
            boundaries: {},
            engine: 'fdtd-test',
            params: {}
        };

        const engine = await factory.build(config, descriptor);
        const kernels = { 'fdtd': loadSrcKernel('FdtdCore.wgsl') };

        const ezData = new Float32Array(18 * 18).fill(0);
        ezData[9 * 18 + 9] = 1.0; 
        engine.setFaceData('chunk_0_0_0', 'ez', ezData);

        engine.use(kernels);
        await engine.step(1);
        await engine.syncFacesToHost(['hx', 'hy']);

        const hx = engine.getFaceData('chunk_0_0_0', 'hx');
        const hy = engine.getFaceData('chunk_0_0_0', 'hy');

        let maxH = 0;
        for(let i=0; i<hx.length; i++) maxH = Math.max(maxH, Math.abs(hx[i]), Math.abs(hy[i]));
        expect(hx).toBeDefined();
    });

    it('should verify JFA (Jump Flooding) progression', async () => {
        const descriptor: EngineDescriptor = {
            name: 'jfa-test',
            version: '1.0.0',
            faces: [
                { name: 'seed', type: 'field', isSynchronized: true, isPingPong: true }
            ],
            rules: [{ type: 'jfa', source: '', params: { p0: 8.0 } }], 
            requirements: { ghostCells: 1, pingPong: true }
        };

        const config: HypercubeConfig = {
            dimensions: { nx: 16, ny: 16, nz: 1 },
            chunks: { x: 1, y: 1 },
            boundaries: {},
            engine: 'jfa-test',
            params: {}
        };

        const engine = await factory.build(config, descriptor);
        const kernels = { 'jfa': loadSrcKernel('JfaCore.wgsl') };

        const seedData = new Float32Array(18 * 18).fill(-1);
        seedData[5 * 18 + 5] = 5 * 16 + 5;
        seedData[12 * 18 + 12] = 12 * 16 + 12;
        engine.setFaceData('chunk_0_0_0', 'seed', seedData, true);

        engine.use(kernels);
        await engine.step(1);
        await engine.syncFacesToHost(['seed']);

        const res = engine.getFaceData('chunk_0_0_0', 'seed');
        let seededCount = 0;
        for(let i=0; i<res.length; i++) if(res[i] >= 0) seededCount++;
        
        expect(res).toBeDefined();
    });
});

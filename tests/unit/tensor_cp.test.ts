import { describe, it, expect, beforeAll, vi } from 'vitest';
import { GpuCoreFactory } from '../../src/GpuCoreFactory';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';
import * as fs from 'fs';
import * as path from 'path';

// Global WebGPU Mocks for Node/Vitest
(globalThis as unknown as { GPUBufferUsage: Record<string, number> }).GPUBufferUsage = {
    MAP_READ: 0x0001, MAP_WRITE: 0x0002, COPY_SRC: 0x0004, COPY_DST: 0x0008,
    INDEX: 0x0010, VERTEX: 0x0020, UNIFORM: 0x0040, STORAGE: 0x0080,
};

(globalThis as unknown as { GPUMapMode: Record<string, number> }).GPUMapMode = {
    READ: 0x0001,
    WRITE: 0x0002
};

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

describe('Tensor-CP Model Verification', () => {
    let factory: GpuCoreFactory;
    
    beforeAll(async () => {
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

    it('should correctly build Tensor-CP engine and allocate buffers', async () => {
        const manifestPath = path.join(__dirname, '../manifests', 'tensor_cp_audit.json');
        const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
        
        const engine = await factory.build(manifest.config, manifest.engine, mockDevice);
        expect(engine).toBeDefined();
        // Check face count via DataContract descriptor
        expect(engine.vGrid.dataContract.descriptor.faces.length).toBe(6);
        
        // rawBuffer should be allocated
        expect(engine.buffer.rawBuffer.byteLength).toBeGreaterThan(0);
    });

    it('should validate 3D reconstruction kernel logic path', async () => {
        const manifestPath = path.join(__dirname, '../manifests', 'tensor_cp_audit.json');
        const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
        const engine = await factory.build(manifest.config, manifest.engine, mockDevice);
        const kernelSrc = fs.readFileSync(path.join(__dirname, '../../src/kernels/wgsl', 'TensorCpCore.wgsl'), 'utf-8');

        // This test validates that the engine can accept the Tensor-CP rules
        engine.use({ 'TensorCpCore': kernelSrc });
        await engine.step(1);
        
        expect(engine.parityManager.currentTick).toBe(1);
    });
});

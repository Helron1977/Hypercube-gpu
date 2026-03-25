import { describe, it, expect } from 'vitest';
import { GpuDispatcher } from '../../src/dispatchers/GpuDispatcher';
import { VirtualGrid } from '../../src/topology/VirtualGrid';
import { ParityManager } from '../../src/ParityManager';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';
import { MasterBuffer } from '../../src/memory/MasterBuffer';
import { IVirtualGrid } from '../../src/topology/GridAbstractions';
import { HypercubeConfig, EngineDescriptor } from '../../src/types';

(globalThis as unknown as { GPUBufferUsage: Record<string, number> }).GPUBufferUsage = {
    MAP_READ: 0x0001, MAP_WRITE: 0x0002, COPY_SRC: 0x0004, COPY_DST: 0x0008,
    INDEX: 0x0010, VERTEX: 0x0020, UNIFORM: 0x0040, STORAGE: 0x0080,
    INDIRECT: 0x0100, QUERY_RESOLVE: 0x0200,
};

describe('LBM Mother Model Integration', () => {
    it('should map LBM parameters and faces to uniforms correctly', async () => {
        // Mock GPU Context
        const mockDevice = {
            createShaderModule: () => ({}),
            createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }), 
            createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
            createBuffer: (desc: { size: number }) => ({ destroy: () => {}, size: desc.size }),
            createBindGroup: () => ({}),
            queue: { 
                writeBuffer: (_buf: unknown, _off: unknown, _data: unknown) => {}, 
                submit: () => {} 
            },
            limits: { minUniformBufferOffsetAlignment: 256 },
            createCommandEncoder: () => ({
                beginComputePass: () => ({
                    setPipeline: () => {},
                    setBindGroup: () => {},
                    dispatchWorkgroups: () => {},
                    end: () => {}
                }),
                finish: () => ({})
            })
        } as unknown as GPUDevice;
        HypercubeGPUContext.setDevice(mockDevice);

        const config = {
            dimensions: { nx: 64, ny: 64, nz: 1 },
            chunks: { x: 1, y: 1, z: 1 },
            boundaries: { all: { role: 'wall' } }
        };

        const lbmDescriptor = {
            name: 'lbm-engine',
            requirements: { ghostCells: 1, pingPong: true },
            faces: [
                { name: 'rho', type: 'scalar', isSynchronized: true },
                { name: 'vx', type: 'scalar', isSynchronized: true },
                { name: 'vy', type: 'scalar', isSynchronized: true },
                { name: 'smoke', type: 'scalar', isSynchronized: true },
                { name: 'f_core', type: 'population', isSynchronized: true } // f-base
            ],
            rules: [{
                type: 'LbmCore',
                params: { p0: 1.8, p1: 0.1 }
            }]
        };

        const vGrid = new VirtualGrid(config as unknown as HypercubeConfig, lbmDescriptor as unknown as EngineDescriptor);
        const buffer = {
            gpuBuffer: {} as GPUBuffer,
            totalSlotsPerChunk: 10,
            strideFace: 66 * 66
        } as unknown as MasterBuffer;
        const parity = new ParityManager(vGrid.dataContract);

        // Spy on writeBuffer to check uniform data
        let capturedUniformData: { buffer: ArrayBuffer } | null = null;
        (mockDevice.queue as unknown as { writeBuffer: Function }).writeBuffer = (_buf: unknown, _off: unknown, data: { buffer: ArrayBuffer }) => {
            capturedUniformData = data;
        };

        const dispatcher = new GpuDispatcher(vGrid, buffer, parity);

        await dispatcher.dispatch(0.5, { 'LbmCore': 'void main() {}' });

        if (!capturedUniformData) throw new Error('No data captured');
        const bufferToUse = (capturedUniformData as unknown as { buffer: ArrayBuffer }).buffer || capturedUniformData;
        const f32 = new Float32Array(bufferToUse);
        const u32 = new Uint32Array(bufferToUse);

        // Verification of LBM-specific mapping
        expect(f32[8]).toBeCloseTo(1.8);  // p0: omega
        expect(f32[9]).toBeCloseTo(0.1);  // p1: inflow

        // Face mappings (rho is first)
        const rhoIdx = parity.getFaceIndices('rho').read;
        expect(u32[16]).toBe(rhoIdx); // f0: rho
    });
});

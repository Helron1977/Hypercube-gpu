import { describe, it, expect, vi } from 'vitest';
import { GpuDispatcher } from '../../src/dispatchers/GpuDispatcher';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';
import { VirtualGrid } from '../../src/topology/VirtualGrid';

describe('Generic Mother Model Integration', () => {
    // Mock WebGPU Globals
    (globalThis as any).GPUBufferUsage = {
        UNIFORM: 1,
        STORAGE: 2,
        COPY_DST: 4
    };
    // Mock WebGPU Context
    const mockBuffer = { getMappedRange: () => new ArrayBuffer(2048), unmap: () => {} };
    const mockPass = { setPipeline: vi.fn(), setBindGroup: vi.fn(), dispatchWorkgroups: vi.fn(), end: vi.fn() };
    const mockEncoder = { beginComputePass: () => mockPass, finish: () => ({}) };
    const mockDevice = {
        createBuffer: () => ({ ...mockBuffer, destroy: () => {}, size: 2048 }),
        createBindGroup: () => ({}),
        createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
        createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
        createShaderModule: () => ({}),
        createCommandEncoder: () => mockEncoder,
        queue: { writeBuffer: vi.fn(), submit: vi.fn() }
    };
    HypercubeGPUContext.setDevice(mockDevice as any);
    (HypercubeGPUContext as any)._isInitialized = true;
    (HypercubeGPUContext as any).alignToUniform = (n: number) => Math.ceil(n/256)*256;

    const config = { dimensions: { nx: 32, ny: 32, nz: 1 }, chunks: { x: 1, y: 1, z: 1 }, chunkLayout: { x: 32, y: 32, z: 1 } };
    const descriptor = {
        name: 'generic',
        requirements: { ghostCells: 1 },
        faces: [{ name: 'a', isSynchronized: true }, { name: 'b', isSynchronized: true }],
        rules: [
            { type: 'TensorCore', faces: { f0: 'a', f1: 'a', f2: 'b' }, params: { p0: 2.0, p1: 1.0 } },
            { type: 'DiffusionCore', faces: { f0: 'a', f1: 'b' }, params: { p0: 0.1 } }
        ]
    };
    const vGrid = new VirtualGrid(config as any, descriptor as any);
    const mockMB = { gpuBuffer: {}, strideFace: 1024, totalSlotsPerChunk: 2 } as any;
    const mockParity = { getFaceIndices: (n: string) => ({ read: 0, write: 1 }) } as any;

    it('should correctly map TensorCore parameters', async () => {
        const dispatcher = new GpuDispatcher(vGrid as any, mockMB, mockParity, mockDevice as any);
        
        // Mock getPipeline and refreshMetadata if needed
        (dispatcher as any).getPipeline = async () => ({ getBindGroupLayout: () => ({}) });

        await dispatcher.dispatch(0, { 'TensorCore': 'some-wgsl' });

        // Verify that writeBuffer was called
        expect(mockDevice.queue.writeBuffer).toHaveBeenCalled();
        const callArgs = (mockDevice.queue.writeBuffer as any).mock.calls[0];
        const rawData = callArgs[2];
        const bufferToUse = rawData.buffer || rawData;
        const writtenData = new Float32Array(bufferToUse);
        
        // p0 check
        expect(writtenData[8]).toBeCloseTo(2.0);
    });
});

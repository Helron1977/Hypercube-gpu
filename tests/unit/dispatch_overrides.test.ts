import { describe, it, expect, vi } from 'vitest';
import { GpuDispatcher } from '../../src/dispatchers/GpuDispatcher';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';
import { VirtualGrid } from '../../src/topology/VirtualGrid';
import { ParityManager } from '../../src/ParityManager';

describe('GpuDispatcher: Dynamic Parameter Overrides (TDD v5.0.4)', () => {
    // Mock WebGPU Globals
    (globalThis as any).GPUBufferUsage = { UNIFORM: 1, STORAGE: 2, COPY_DST: 4 };

    const mockDevice = {
        createBuffer: () => ({ destroy: () => {}, size: 2048 }),
        createBindGroup: () => ({}),
        createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
        createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
        createShaderModule: () => ({}),
        queue: { writeBuffer: vi.fn(), submit: vi.fn() },
        limits: { minUniformBufferOffsetAlignment: 256 },
        createCommandEncoder: () => ({
            beginComputePass: () => ({ 
                setPipeline: vi.fn(), setBindGroup: vi.fn(), 
                dispatchWorkgroups: vi.fn(), end: vi.fn() 
            }),
            finish: () => ({})
        })
    };

    HypercubeGPUContext.setDevice(mockDevice as any);
    (HypercubeGPUContext as any)._isInitialized = true;
    (HypercubeGPUContext as any).alignToUniform = (n: number) => Math.ceil(n/256)*256;

    const config = { 
        dimensions: { nx: 32, ny: 32, nz: 1 }, 
        chunks: { x: 1, y: 1, z: 1 },
        boundaries: { all: { role: 'wall' } },
        engine: 'v5-test',
        params: { p0: 1.0, p1: 2.0 } // Default values
    };

    const descriptor = {
        name: 'v5-test',
        version: '5.0.4',
        requirements: { ghostCells: 1, pingPong: true },
        faces: [{ name: 'scalar', type: 'scalar', isSynchronized: true }],
        rules: [{ type: 'Update', faces: ['scalar'] }]
    };

    const vGrid = new VirtualGrid(config as any, descriptor as any);
    const mockMB = { gpuBuffer: {}, strideFace: 1024, totalSlotsPerChunk: 10, layout: { totalStandardSlotsPerChunk: 10 } } as any;
    const parity = new ParityManager(vGrid.dataContract);

    it('should override p0 when passed to dispatch()', async () => {
        const dispatcher = new GpuDispatcher(vGrid, mockMB, parity, mockDevice as any);
        
        // This call should fail to compile (TS) or the params won't be applied (Runtime)
        // @ts-ignore - Testing new API before implementation
        await dispatcher.dispatch(0, { 'Update': 'void main() {}' }, { p0: 5.5 });

        expect(mockDevice.queue.writeBuffer).toHaveBeenCalled();
        const callArgs = mockDevice.queue.writeBuffer.mock.calls[0];
        const f32 = new Float32Array(callArgs[2]);

        // Struct Mapping: uniforms at base+8 is p0
        expect(f32[8]).toBe(5.5);
        
        // Original config must remain unchanged
        expect(config.params.p0).toBe(1.0);
    });

    it('should fall back to config params when no override provided', async () => {
        const dispatcher = new GpuDispatcher(vGrid, mockMB, parity, mockDevice as any);
        
        await dispatcher.dispatch(0, { 'Update': 'void main() {}' });

        expect(mockDevice.queue.writeBuffer).toHaveBeenCalled();
        const callArgs = mockDevice.queue.writeBuffer.mock.calls[1]; // Second call
        const f32 = new Float32Array(callArgs[2]);

        expect(f32[8]).toBe(1.0);
    });
});

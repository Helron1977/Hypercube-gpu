import { describe, it, expect, vi } from 'vitest';
import { GpuDispatcher } from '../../src/dispatchers/GpuDispatcher';
import { MasterBuffer } from '../../src/memory/MasterBuffer';
import { VirtualGrid } from '../../src/topology/VirtualGrid';
import { ParityManager } from '../../src/ParityManager';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';

describe('GpuDispatcher: Professional Atomics (TDD v5.0.4)', () => {
    // Mock WebGPU Globals
    (globalThis as any).GPUBufferUsage = { UNIFORM: 1, STORAGE: 2, COPY_DST: 4, COPY_SRC: 8 };

    const mockDevice = {
        createBuffer: vi.fn((desc: any) => ({ 
            destroy: () => {}, 
            size: desc.size, 
            label: desc.label,
            usage: desc.usage
        })),
        createBindGroup: vi.fn(() => ({})),
        createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
        createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
        createShaderModule: vi.fn(),
        queue: { writeBuffer: vi.fn(), submit: vi.fn() },
        limits: { minUniformBufferOffsetAlignment: 256 },
        createCommandEncoder: () => ({
            beginComputePass: () => ({ 
                setPipeline: vi.fn(), setBindGroup: vi.fn(), 
                dispatchWorkgroups: vi.fn(), end: vi.fn() 
            }),
            finish: () => ({}),
            copyBufferToBuffer: vi.fn()
        })
    };

    HypercubeGPUContext.setDevice(mockDevice as any);
    (HypercubeGPUContext as any)._isInitialized = true;
    (HypercubeGPUContext as any).alignToUniform = (n: number) => Math.ceil(n/256)*256;

    it('should identify atomic faces and generate CAS loop for atomic_f32', async () => {
        const descriptor = {
            name: 'atomic-test',
            faces: [
                { name: 'weights', type: 'atomic_f32', isSynchronized: false },
                { name: 'counter', type: 'atomic_u32', isSynchronized: false }
            ],
            rules: [{ type: 'Update', faces: ['weights', 'counter'] }],
            requirements: { ghostCells: 1, pingPong: false }
        };

        const config = { 
            dimensions: { nx: 32, ny: 32, nz: 1 }, 
            chunks: { x: 1, y: 1, z: 1 },
            boundaries: { all: { role: 'wall' } },
            engine: 'atomic-test',
            params: {}
        };

        const vGrid = new VirtualGrid(config as any, descriptor as any);
        const parity = new ParityManager(vGrid.dataContract);
        const mockMB = new MasterBuffer(vGrid, mockDevice as any);
        const dispatcher = new GpuDispatcher(vGrid, mockMB, parity, mockDevice as any);

        const { code: header } = dispatcher.getWgslHeader('Update');

        // 1. Verify Binding 2 exists in header
        expect(header).toContain('@group(0) @binding(2) var<storage, read_write> dataAtomic: array<atomic<i32>>');

        // 2. Verify CAS loop helper exists
        expect(header).toContain('fn _hypercube_atomicAddF32');
        expect(header).toContain('atomicCompareExchangeWeak');

        // 3. Verify Atomic Macros
        expect(header).toContain('fn atomicAdd_weights');
        expect(header).toContain('fn atomicAdd_counter');
        
        // 4. Verify routing to dataAtomic
        expect(header).toContain('dataAtomic[');
    });

    it('should allocate a separate atomic buffer in MasterBuffer', () => {
        const descriptor = {
            faces: [{ name: 'a', type: 'atomic_f32', isSynchronized: false }],
            requirements: { ghostCells: 0, pingPong: false }
        };
        const config = { dimensions: { nx: 8, ny: 8, nz: 1 }, chunks: { x: 1, y: 1 }, engine: 't', params: {} };
        const vGrid = new VirtualGrid(config as any, descriptor as any);

        const mb = new MasterBuffer(vGrid, mockDevice as any);
        
        // Should have allocated standard gpuBuffer AND atomicBuffer
        expect((mb as any).gpuAtomicBuffer).toBeDefined();
        expect(mockDevice.createBuffer).toHaveBeenCalledWith(expect.objectContaining({
            label: expect.stringContaining('Atomic')
        }));
    });
});

import { describe, it, expect, vi } from 'vitest';
import { GpuDispatcher } from '../../src/dispatchers/GpuDispatcher';
import { VirtualGrid } from '../../src/topology/VirtualGrid';
import { ParityManager } from '../../src/ParityManager';

describe('GpuDispatcher: Header Generation Logic', () => {
    
    const mockMB = { 
        gpuBuffer: {}, strideFace: 1024, standardByteLength: 4096, 
        atomicByteLength: 0, gpuAtomicBuffer: null,
        layout: { getFaceOffset: () => 0, totalStandardSlotsPerChunk: 1, totalAtomicSlotsPerChunk: 0 }
    } as any;

    const mockDevice = {
        limits: { minUniformBufferOffsetAlignment: 256 },
        createBuffer: vi.fn(),
        createBindGroup: vi.fn(),
        createComputePipeline: vi.fn(),
        createComputePipelineAsync: vi.fn(),
        createShaderModule: vi.fn()
    };

    it('should NOT inject Binding 2 or CAS loop for purely standard engines', () => {
        const descriptor = {
            faces: [{ name: 'scalar', type: 'scalar' }],
            rules: [{ type: 'Update', faces: ['scalar'] }],
            requirements: { ghostCells: 0 }
        };
        const vGrid = new VirtualGrid({ dimensions: { nx: 8, ny: 8 }, chunks: { x: 1, y: 1 }, engine: 'std' } as any, descriptor as any);
        const dispatcher = new GpuDispatcher(vGrid, mockMB, new ParityManager(vGrid.dataContract), mockDevice as any);

        const { code: header } = dispatcher.getWgslHeader('Update');
        
        expect(header).not.toContain('@binding(2)');
        expect(header).not.toContain('_hypercube_atomicAddF32');
        expect(header).toContain('@binding(0) var<storage, read_write> data: array<f32>;');
    });

    it('should inject Binding 2 but NOT CAS loop for purely atomic_u32 engines', () => {
        const descriptor = {
            faces: [{ name: 'counter', type: 'atomic_u32' }],
            rules: [{ type: 'Update', faces: ['counter'] }],
            requirements: { ghostCells: 0 }
        };
        const vGrid = new VirtualGrid({ dimensions: { nx: 8, ny: 8 }, chunks: { x: 1, y: 1 }, engine: 'u32' } as any, descriptor as any);
        const dispatcher = new GpuDispatcher(vGrid, mockMB, new ParityManager(vGrid.dataContract), mockDevice as any);

        const { code: header } = dispatcher.getWgslHeader('Update');
        
        expect(header).toContain('@binding(2)');
        expect(header).not.toContain('_hypercube_atomicAddF32');
        expect(header).toContain('fn atomicAdd_counter');
        expect(header).toContain('atomicAdd(&dataAtomic');
    });

    it('should inject BOTH Binding 2 and CAS loop for atomic_f32 engines', () => {
        const descriptor = {
            faces: [{ name: 'flux', type: 'atomic_f32' }],
            rules: [{ type: 'Update', faces: ['flux'] }],
            requirements: { ghostCells: 0 }
        };
        const vGrid = new VirtualGrid({ dimensions: { nx: 8, ny: 8 }, chunks: { x: 1, y: 1 }, engine: 'f32' } as any, descriptor as any);
        const dispatcher = new GpuDispatcher(vGrid, mockMB, new ParityManager(vGrid.dataContract), mockDevice as any);

        const { code: header } = dispatcher.getWgslHeader('Update');
        
        expect(header).toContain('@binding(2)');
        expect(header).toContain('_hypercube_atomicAddF32');
        expect(header).toContain('fn atomicAdd_flux');
        expect(header).toContain('_hypercube_atomicAddF32(');
    });
});

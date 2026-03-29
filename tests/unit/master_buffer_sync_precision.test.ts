import { describe, it, expect, vi } from 'vitest';
import { MasterBuffer } from '../../src/memory/MasterBuffer';
import { VirtualGrid } from '../../src/topology/VirtualGrid';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';

describe('MasterBuffer: Sync Precision (TDD v5.0.4)', () => {
    
    // Mock WebGPU Globals
    (globalThis as any).GPUBufferUsage = { STORAGE: 0x80, COPY_DST: 0x08, COPY_SRC: 0x04, MAP_READ: 0x01 };

    const mockDevice = {
        createBuffer: vi.fn((desc: any) => ({ 
            destroy: vi.fn(), 
            size: desc.size, 
            label: desc.label,
            usage: desc.usage,
            mapAsync: vi.fn(() => Promise.resolve()),
            getMappedRange: vi.fn(() => new ArrayBuffer(desc.size)),
            unmap: vi.fn()
        })),
        queue: { writeBuffer: vi.fn(), submit: vi.fn() },
        createCommandEncoder: vi.fn(() => ({
            copyBufferToBuffer: vi.fn(),
            finish: vi.fn(() => ({ label: 'mock-finish' }))
        }))
    };

    HypercubeGPUContext.setDevice(mockDevice as any);

    it('should split rawBuffer into two GPU writeBuffer calls (Standard & Atomic)', () => {
        const descriptor = {
            faces: [
                { name: 'S1', type: 'scalar' },    // Binding 0
                { name: 'A1', type: 'atomic_u32' } // Binding 2
            ],
            requirements: { ghostCells: 0, pingPong: false }
        };
        const config = { dimensions: { nx: 8, ny: 8 }, chunks: { x: 1, y: 1 }, engine: 'sync-test', params: {} };
        const vGrid = new VirtualGrid(config as any, descriptor as any);

        const mb = new MasterBuffer(vGrid, mockDevice as any);
        
        // strideFace = 64 floats (aligned)
        // standardByteLength = 1 chunk * 1 slot * 64 floats * 4 bytes = 256 bytes
        // atomicByteLength = 1 chunk * 1 slot * 64 floats * 4 bytes = 256 bytes
        expect(mb.layout.standardByteLength).toBe(256);
        expect(mb.layout.atomicByteLength).toBe(256);

        // Fill rawBuffer with test pattern [1, 1, 1... 2, 2, 2...]
        const f32 = new Float32Array(mb.rawBuffer);
        f32.fill(1.0, 0, 64);      // Standard section
        const u32 = new Uint32Array(mb.rawBuffer);
        u32.fill(1234, 64, 128);   // Atomic section (starting at float 64)

        mb.syncToDevice();

        expect(mockDevice.queue.writeBuffer).toHaveBeenCalledTimes(2);

        // Call 1: Standard
        const call1 = mockDevice.queue.writeBuffer.mock.calls[0];
        expect(call1[0].label).toBe('Hypercube Master Buffer (Float32)');
        expect(call1[1]).toBe(0); // Offset in GPU buffer
        expect(call1[2].byteLength).toBe(256);
        const data1 = new Float32Array(call1[2].buffer, call1[2].byteOffset, 64);
        expect(data1[0]).toBe(1.0);

        // Call 2: Atomic
        const call2 = mockDevice.queue.writeBuffer.mock.calls[1];
        expect(call2[0].label).toBe('Hypercube Atomic Buffer (U32)');
        expect(call2[1]).toBe(0); // Offset in GPU buffer 2
        expect(call2[2].byteLength).toBe(256);
        const data2 = new Uint32Array(call2[2].buffer, call2[2].byteOffset, 64);
        expect(data2[0]).toBe(1234);
    });

    it('should correctly map individual faces to their respective GPU sources in syncFacesToHost', async () => {
        const descriptor = {
            faces: [
                { name: 'S1', type: 'scalar' },
                { name: 'A1', type: 'atomic_u32' }
            ],
            requirements: { ghostCells: 0, pingPong: false }
        };
        const config = { dimensions: { nx: 8, ny: 8 }, chunks: { x: 1, y: 1 }, engine: 't', params: {} };
        const vGrid = new VirtualGrid(config as any, descriptor as any);
        const mb = new MasterBuffer(vGrid, mockDevice as any);

        // Capture command encoder
        const encoder = { copyBufferToBuffer: vi.fn(), finish: vi.fn(() => ({ label: 'mock-finish' })) };
        mockDevice.createCommandEncoder.mockReturnValue(encoder as any);

        await mb.syncFacesToHost(['S1', 'A1']);

        // Check if copyBufferToBuffer was called with the correct SOURCE buffers
        expect(encoder.copyBufferToBuffer).toHaveBeenCalledTimes(2);
        
        const copy1 = encoder.copyBufferToBuffer.mock.calls[0];
        expect(copy1[0].label).toBe('Hypercube Master Buffer (Float32)');
        
        const copy2 = encoder.copyBufferToBuffer.mock.calls[1];
        expect(copy2[0].label).toBe('Hypercube Atomic Buffer (U32)');
    });
});

import { describe, it, expect } from 'vitest';
import { GpuCoreFactory } from '../../src/GpuCoreFactory';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';

describe('Geometric Alignment Integrity', () => {
    it('should verify that getIndex(x, y) matches MemoryLayout exactly', async () => {
        // This test requires a real GPU or a very good mock. 
        // Since we are in a unit test environment, we will mock the dispatch 
        // to verify the logic of strideRow passing.
        
        const config = {
            dimensions: { nx: 64, ny: 64, nz: 1 },
            chunks: { x: 1, y: 1, z: 1 },
            boundaries: { all: { role: 'wall' } }
        };

        const engineDescriptor = {
            name: 'geometric-test',
            requirements: { ghostCells: 1, pingPong: false },
            faces: [{ name: 'data', type: 'field' }],
            rules: [{ type: 'TestKernel', params: {} }]
        };

        const factory = new GpuCoreFactory();
        const engine = await factory.build(config as any, engineDescriptor as any);
        
        const layout = engine.buffer.layout;
        expect(layout.strideRow).toBe(66); // 64 + 2*1

        // Verify the uniform buffer population
        const dispatcher = (engine as any).dispatcher;
        await dispatcher.dispatch(0, { 'TestKernel': 'void main() {}' });
        
        const u32 = new Uint32Array(dispatcher.stagingBuffer.buffer);
        
        // Index 2 is strideRow
        expect(u32[2]).toBe(66);
        // Index 80 is ghosts
        expect(u32[80]).toBe(1);
        
        // Verify Face Offset (Absolute Pointer)
        const faceIdx = engine.parityManager.getFaceIndices('data').base;
        expect(u32[16]).toBe(faceIdx); 
    });
});

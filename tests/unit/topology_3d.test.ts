import { describe, it, expect } from 'vitest';
import { MapConstructor } from '../../src/topology/MapConstructor';

describe('3D Topology Validation', () => {
    it('should generate exactly 26 joints for a chunk in a 3D periodic grid', () => {
        const mc = new MapConstructor();
        const dims = { nx: 32, ny: 32, nz: 32 };
        const chunks = { x: 2, y: 2, z: 2 };
        const boundaries = { all: { role: 'periodic' } };

        const virtualChunks = mc.buildMap(dims, chunks, boundaries as any);

        expect(virtualChunks.length).toBe(8);
        
        for (const chunk of virtualChunks) {
            expect(chunk.joints.length).toBe(26);
            
            // Verify unique faces
            const faceNames = chunk.joints.map(j => j.face);
            const uniqueFaces = new Set(faceNames);
            expect(uniqueFaces.size).toBe(26);

            // Verify specific 3D directions exist
            expect(faceNames).toContain('front');
            expect(faceNames).toContain('back');
            expect(faceNames).toContain('front-top-left');
            expect(faceNames).toContain('back-bottom-right');
            expect(faceNames).toContain('front-left');
        }
    });

    it('should handle wall boundaries correctly in 3D', () => {
        const mc = new MapConstructor();
        const dims = { nx: 32, ny: 32, nz: 32 };
        const chunks = { x: 1, y: 1, z: 2 };
        const boundaries = { front: { role: 'wall' }, back: { role: 'wall' } };

        const virtualChunks = mc.buildMap(dims, chunks, boundaries as any);

        // Chunk 0 is at front (cz=0)
        const frontChunk = virtualChunks.find(c => c.z === 0)!;
        const frontJoint = frontChunk.joints.find(j => j.face === 'front');
        expect(frontJoint?.role).toBe('wall');

        // Chunk 1 is at back (cz=1)
        const backChunk = virtualChunks.find(c => c.z === 1)!;
        const backJoint = backChunk.joints.find(j => j.face === 'back');
        expect(backJoint?.role).toBe('wall');
    });
});

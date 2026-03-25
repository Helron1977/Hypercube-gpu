import { describe, it, expect } from 'vitest';
import { TopologyResolver, BoundaryRoleID } from '../../src/topology/TopologyResolver';
import { VirtualChunk } from '../../src/topology/GridAbstractions';

describe('TopologyResolver', () => {
    it('should resolve global boundaries correctly', () => {
        const resolver = new TopologyResolver();
        const vChunk: VirtualChunk = {
            id: 'c0',
            x: 0, y: 0, z: 0,
            localDimensions: { nx: 128, ny: 128, nz: 1 },
            joints: []
        };
        const globalBoundaries = {
            left: { role: 'inflow' },
            right: { role: 'outflow' },
            top: { role: 'wall' },
            bottom: { role: 'wall' }
        };

        const topo = resolver.resolve(vChunk, { x: 1, y: 1 }, globalBoundaries as any);
        expect(topo.leftRole).toBe(BoundaryRoleID.INFLOW);
        expect(topo.rightRole).toBe(BoundaryRoleID.OUTFLOW);
        expect(topo.topRole).toBe(BoundaryRoleID.WALL);
        expect(topo.bottomRole).toBe(BoundaryRoleID.WALL);
    });

    it('should prioritize joints over global boundaries', () => {
        const resolver = new TopologyResolver();
        const vChunk: VirtualChunk = {
            id: 'c0',
            x: 0, y: 0, z: 0,
            localDimensions: { nx: 128, ny: 128, nz: 1 },
            joints: [
                { face: 'right', role: 'joint', neighborId: 'c1' }
            ]
        };
        const globalBoundaries = { all: { role: 'wall' } };

        const topo = resolver.resolve(vChunk, { x: 2, y: 1 }, globalBoundaries as any);
        expect(topo.rightRole).toBe(BoundaryRoleID.CONTINUITY);
        expect(topo.leftRole).toBe(BoundaryRoleID.WALL);
    });
});

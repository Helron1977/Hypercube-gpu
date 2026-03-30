import { describe, it, expect, beforeEach } from 'vitest';
import { ParityManager } from '../../src/ParityManager';
import { DataContract } from '../../src/DataContract';
import { EngineDescriptor } from '../../src/types';

describe('Triple Buffering (Modulo > 2)', () => {
    const descriptor: EngineDescriptor = {
        name: 'test-wave',
        version: '1.0.0',
        requirements: { ghostCells: 1, pingPong: true },
        faces: [
            { name: 'phi', type: 'scalar', isSynchronized: true, numSlots: 3 },
            { name: 'vel', type: 'vector', isSynchronized: true }
        ],
        rules: []
    };

    const dataContract = new DataContract(descriptor);
    let parityManager: ParityManager;

    beforeEach(() => {
        parityManager = new ParityManager(dataContract);
    });

    it('should calculate correct number of slots in DataContract', () => {
        const mappings = dataContract.getFaceMappings();
        const phi = mappings.find(m => m.name === 'phi')!;
        const vel = mappings.find(m => m.name === 'vel')!;

        expect(phi.numSlots).toBe(3); 
        expect(vel.numSlots).toBe(2); 
    });

    it('should allocate correct memory for triple buffering', () => {
        // 10x10 grid with 1 ghost cell = 12x12 = 144 cells
        // phi: 1 component * 3 slots = 3
        // vel: 3 components * 2 slots = 6
        // total slots: 9
        // bytes: 144 * 4 * 9 = 5184
        const bytes = dataContract.calculateChunkBytes(10, 10, 1, 1);
        expect(bytes).toBe(12 * 12 * 4 * 9);
    });

    it('should cycle through 3 slots for phi (Dynamic Rotation)', () => {
        // Tick 0 (Now=0, Next=1)
        let indices = parityManager.getFaceIndices('phi');
        expect(indices.read).toBe(0);
        expect(indices.write).toBe(1);

        // Tick 1 (Now=1, Next=2)
        parityManager.increment();
        indices = parityManager.getFaceIndices('phi');
        expect(indices.read).toBe(1);
        expect(indices.write).toBe(2);
    });

    it('should cycle through 2 slots for vel (Dynamic Rotation)', () => {
        // vel starts after phi (3 slots)
        const base = 3;
        const comp = 3;

        // Tick 0 (Now=base+0, Next=base+comp)
        let indices = parityManager.getFaceIndices('vel');
        expect(indices.read).toBe(base); 
        expect(indices.write).toBe(base + comp);

        // Tick 1 (Now=base+comp, Next=base+0)
        parityManager.increment();
        indices = parityManager.getFaceIndices('vel');
        expect(indices.read).toBe(base + comp); // 3 + 3 = 6
        expect(indices.write).toBe(base);       // 3 + 0 = 3
    });
});

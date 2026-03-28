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
        expect(vel.numSlots).toBe(2); // Default because pingPong requirement is true
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

    it('should cycle through 3 slots for phi', () => {
        // Tick 0
        let indices = parityManager.getFaceIndices('phi');
        expect(indices.read).toBe(0);
        expect(indices.write).toBe(1);

        // Tick 1
        parityManager.increment();
        indices = parityManager.getFaceIndices('phi');
        expect(indices.read).toBe(1);
        expect(indices.write).toBe(2);

        // Tick 2
        parityManager.increment();
        indices = parityManager.getFaceIndices('phi');
        expect(indices.read).toBe(2);
        expect(indices.write).toBe(0);

        // Tick 3 (back to 0)
        parityManager.increment();
        indices = parityManager.getFaceIndices('phi');
        expect(indices.read).toBe(0);
        expect(indices.write).toBe(1);
    });

    it('should still cycle through 2 slots for vel', () => {
        // vel starts after phi (3 slots * 1 component = 3)
        const base = 3;

        // Tick 0
        let indices = parityManager.getFaceIndices('vel');
        expect(indices.read).toBe(base + 0);
        expect(indices.write).toBe(base + 3);

        // Tick 1
        parityManager.increment();
        indices = parityManager.getFaceIndices('vel');
        expect(indices.read).toBe(base + 3);
        expect(indices.write).toBe(base + 0);

        // Tick 2 (back to 0 for modulo 2)
        parityManager.increment();
        indices = parityManager.getFaceIndices('vel');
        expect(indices.read).toBe(base + 0);
        expect(indices.write).toBe(base + 3);
    });
});

import { DataContract } from './DataContract';

export interface FaceIndices {
    base: number;
    read: number;
    write: number;
    old: number;
    slotRead: number;
    slotWrite: number;
    slotOld: number;
}

/**
 * Manages the Dynamic Temporal Rotation for simulation faces.
 * Ensures the logical states (Now, Next, Old) map to rotating physical slots.
 */
export class ParityManager {
    public currentTick: number = 0;
    private faceIndexCache: Map<string, FaceIndices> = new Map();

    constructor(private dataContract: DataContract) {
        this.updateCache();
    }

    public increment(): void {
        this.currentTick++;
        this.updateCache();
    }

    public getFaceIndices(faceName: string): FaceIndices {
        const indices = this.faceIndexCache.get(faceName);
        if (!indices) {
            throw new Error(`ParityManager: Face '${faceName}' not found.`);
        }
        return indices;
    }

    private updateCache(): void {
        const mappings = this.dataContract.getFaceMappings();

        for (const m of mappings) {
            const numSlots = m.numSlots || 1;
            
            if (numSlots > 1) {
                // PURE MATHEMATICAL ROTATION (v6.0 Universal Standard)
                // Now: The current physical slot for this tick.
                // Next: The next physical slot (destination for write).
                // Old: The previous physical slot.
                const tick = this.currentTick % numSlots;
                
                const nowSlot = tick;
                const nextSlot = (tick + 1) % numSlots;
                const oldSlot = (tick - 1 + numSlots) % numSlots;

                const comp = m.numComponents || 1;

                this.faceIndexCache.set(m.name, {
                    base: m.pointerOffset,
                    read: m.pointerOffset + nowSlot * comp,
                    write: m.pointerOffset + nextSlot * comp,
                    old: m.pointerOffset + oldSlot * comp,
                    slotRead: nowSlot,
                    slotWrite: nextSlot,
                    slotOld: oldSlot
                });
            } else {
                // Static/Single-slot face
                this.faceIndexCache.set(m.name, {
                    base: m.pointerOffset,
                    read: m.pointerOffset,
                    write: m.pointerOffset,
                    old: m.pointerOffset,
                    slotRead: 0,
                    slotWrite: 0,
                    slotOld: 0
                });
            }
        }
    }
}

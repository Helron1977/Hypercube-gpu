import { DataContract } from './DataContract';

/**
 * Manages the "Ping-Pong" double buffering state for a set of data faces.
 * Ensures the GPU reads from one buffer while writing to the other.
 */
export class ParityManager {
    public currentTick: number = 0;
    private faceIndexCache: Map<string, { base: number; read: number; write: number; old: number }> = new Map();

    constructor(private dataContract: DataContract) {
        this.updateCache();
    }

    public increment(): void {
        this.currentTick++;
        this.updateCache();
    }

    public getFaceIndices(faceName: string, modulo?: number): { base: number; read: number; write: number; old: number } {
        const indices = this.faceIndexCache.get(faceName);
        if (!indices) {
            throw new Error(`ParityManager: Face '${faceName}' not found.`);
        }
        
        // Use the cached indices which are pre-calculated for the correct numSlots
        return indices;
    }

    private updateCache(): void {
        const mappings = this.dataContract.getFaceMappings();
        let absolutePointer = 0;

        for (const m of mappings) {
            if (m.numSlots > 1) {
                const tick = this.currentTick % m.numSlots;
                this.faceIndexCache.set(m.name, {
                    base: absolutePointer,
                    read: absolutePointer + (tick * m.numComponents),
                    write: absolutePointer + (((tick + 1) % m.numSlots) * m.numComponents),
                    old: absolutePointer + (((tick - 1 + m.numSlots) % m.numSlots) * m.numComponents)
                });
                absolutePointer += m.numSlots * m.numComponents;
            } else {
                this.faceIndexCache.set(m.name, {
                    base: absolutePointer,
                    read: absolutePointer,
                    write: absolutePointer,
                    old: absolutePointer
                });
                absolutePointer += m.numComponents;
            }
        }
    }
}

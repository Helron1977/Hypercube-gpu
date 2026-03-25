import { DataContract } from './DataContract';

/**
 * Manages the "Ping-Pong" double buffering state for a set of data faces.
 * Ensures the GPU reads from one buffer while writing to the other.
 */
export class ParityManager {
    public currentTick: number = 0;
    private faceIndexCache: Map<string, { base: number; read: number; write: number }> = new Map();

    constructor(private dataContract: DataContract) {
        this.updateCache();
    }

    public increment(): void {
        this.currentTick++;
        this.updateCache();
    }

    public getFaceIndices(faceName: string): { base: number; read: number; write: number } {
        const indices = this.faceIndexCache.get(faceName);
        if (!indices) {
            throw new Error(`ParityManager: Face '${faceName}' not found.`);
        }
        return indices;
    }

    private updateCache(): void {
        const mappings = this.dataContract.getFaceMappings();
        let absolutePointer = 0;

        for (const m of mappings) {
            if (m.isPingPong) {
                // Alternates between [pointer, pointer+numComponents] and [pointer+numComponents, pointer]
                const parity = this.currentTick % 2;
                this.faceIndexCache.set(m.name, {
                    base: absolutePointer,
                    read: absolutePointer + (parity * m.numComponents),
                    write: absolutePointer + ((1 - parity) * m.numComponents)
                });
                absolutePointer += 2 * m.numComponents;
            } else {
                this.faceIndexCache.set(m.name, {
                    base: absolutePointer,
                    read: absolutePointer,
                    write: absolutePointer
                });
                absolutePointer += m.numComponents;
            }
        }
    }
}

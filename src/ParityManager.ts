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

    public getFaceIndices(faceName: string, modulo: number = 2): { base: number; read: number; write: number } {
        const indices = this.faceIndexCache.get(faceName);
        if (!indices) {
            throw new Error(`ParityManager: Face '${faceName}' not found.`);
        }
        
        // If the user requests a different modulo (e.g. 3 for WAVE), we override the cache logic
        if (modulo > 2) {
            const mappings = this.dataContract.getFaceMappings();
            const m = mappings.find(f => f.name === faceName);
            if (m && m.isPingPong) {
                const tick = this.currentTick % modulo;
                return {
                    base: indices.base,
                    read: indices.base + (tick * m.numComponents),
                    write: indices.base + (((tick + 1) % modulo) * m.numComponents)
                };
            }
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

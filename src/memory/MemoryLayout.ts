import { IVirtualGrid } from '../topology/GridAbstractions';
import { DataContract } from '../DataContract';

/**
 * MemoryLayout handles the logical partitioning of the simulation data.
 * It is agnostic to the underlying storage (CPU vs GPU).
 */
export class MemoryLayout {
    public readonly byteLength: number;
    public readonly strideFace: number; // in float32 (4-byte) units
    public readonly strideRow: number;  // in float32 units
    public readonly totalStandardSlotsPerChunk: number;
    public readonly totalAtomicSlotsPerChunk: number;
    public readonly standardByteLength: number;
    public readonly atomicByteLength: number;
    public readonly faceMappings: any[];
    public readonly cellsPerFaceRaw: number;

    constructor(private vGrid: IVirtualGrid) {
        const dataContract = this.vGrid.dataContract;
        this.faceMappings = dataContract.getFaceMappings();

        let maxNx = 0, maxNy = 0, maxNz = 0;
        for (const chunk of this.vGrid.chunks) {
            maxNx = Math.max(maxNx, chunk.localDimensions.nx);
            maxNy = Math.max(maxNy, chunk.localDimensions.ny);
            maxNz = Math.max(maxNz, chunk.localDimensions.nz);
        }

        const ghosts = dataContract.descriptor.requirements.ghostCells;
        const pNx = maxNx + 2 * ghosts;
        const pNy = maxNy + 2 * ghosts;
        const pNz = maxNz > 1 ? maxNz + 2 * ghosts : 1;
        this.strideRow = pNx;
        this.cellsPerFaceRaw = pNx * pNy * pNz;

        const bytesPerFaceRaw = this.cellsPerFaceRaw * 4;
        const bytesPerFaceAligned = Math.ceil(bytesPerFaceRaw / 256) * 256;
        this.strideFace = bytesPerFaceAligned / 4;

        this.totalStandardSlotsPerChunk = this.faceMappings
            .filter(f => !(f.type || 'scalar').startsWith('atomic'))
            .reduce((acc, f) => acc + (f.numComponents * f.numSlots), 0);
            
        this.totalAtomicSlotsPerChunk = this.faceMappings
            .filter(f => (f.type || 'scalar').startsWith('atomic'))
            .reduce((acc, f) => acc + (f.numComponents * f.numSlots), 0);

        this.standardByteLength = this.vGrid.chunks.length * this.totalStandardSlotsPerChunk * bytesPerFaceAligned;
        this.atomicByteLength = this.vGrid.chunks.length * this.totalAtomicSlotsPerChunk * bytesPerFaceAligned;
        
        // Total logical length (sum of both)
        this.byteLength = this.standardByteLength + this.atomicByteLength;
    }

    /**
     * Identifies which hardware binding a face belongs to.
     */
    public getBinding(faceIdx: number): 0 | 2 {
        const type = this.faceMappings[faceIdx].type || 'scalar';
        return type.startsWith('atomic') ? 2 : 0;
    }

    /**
     * Calculates the float32 offset within the specific buffer (standard or atomic).
     */
    public getFaceOffset(chunkIdx: number, faceIdx: number, slotIdx: number = 0): number {
        const binding = this.getBinding(faceIdx);
        const slotsPerChunk = (binding === 0) ? this.totalStandardSlotsPerChunk : this.totalAtomicSlotsPerChunk;
        
        let offset = chunkIdx * slotsPerChunk;
        for (let i = 0; i < faceIdx; i++) {
            if (this.getBinding(i) === binding) {
                offset += this.faceMappings[i].numComponents * this.faceMappings[i].numSlots;
            }
        }
        offset += slotIdx * this.faceMappings[faceIdx].numComponents;
        return offset * this.strideFace;
    }

    /**
     * Calculates the offset in the logical rawBuffer (CPU side).
     * Standard faces at [0, standardByteLength), Atomics at [standardByteLength, total).
     */
    public getLogicalFaceOffset(chunkIdx: number, faceIdx: number, slotIdx: number = 0): number {
        const binding = this.getBinding(faceIdx);
        const bufferOffset = (binding === 0) ? 0 : this.standardByteLength / 4;
        return bufferOffset + this.getFaceOffset(chunkIdx, faceIdx, slotIdx);
    }

    /** @deprecated - Use getFaceOffset and getBinding */
    public get totalSlotsPerChunk(): number {
        return this.totalStandardSlotsPerChunk + this.totalAtomicSlotsPerChunk;
    }
}

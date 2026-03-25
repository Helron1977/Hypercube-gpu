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
    public readonly totalSlotsPerChunk: number;
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

        this.totalSlotsPerChunk = this.faceMappings.reduce((acc, f) => acc + f.numComponents * (f.isPingPong ? 2 : 1), 0);
        this.byteLength = this.vGrid.chunks.length * this.totalSlotsPerChunk * bytesPerFaceAligned;
    }

    /**
     /**
     * Calculates the float32 offset for a specific chunk and face.
     */
    public getFaceOffset(chunkIdx: number, faceIdx: number, isBackBuffer: boolean = false): number {
        let offset = chunkIdx * this.totalSlotsPerChunk;
        for (let i = 0; i < faceIdx; i++) {
            offset += this.faceMappings[i].numComponents * (this.faceMappings[i].isPingPong ? 2 : 1);
        }
        if (isBackBuffer) offset += this.faceMappings[faceIdx].numComponents;
        return offset * this.strideFace;
    }
}

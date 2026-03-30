import { IVirtualGrid } from '../topology/GridAbstractions';

/**
 * MemoryLayout handles the logical partitioning of the simulation data.
 * v6.0: Now supports 'Standard Grid Faces' and 'Global Workspace' (scalars/reductions).
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
    public readonly globalMappings: any[];
    public readonly cellsPerFaceRaw: number;

    constructor(private vGrid: IVirtualGrid) {
        const dataContract = this.vGrid.dataContract;
        this.faceMappings = dataContract.getFaceMappings();
        this.globalMappings = dataContract.getGlobalMappings();

        let maxNx = 0, maxNy = 0, maxNz = 0;
        for (const chunk of this.vGrid.chunks) {
            maxNx = Math.max(maxNx, chunk.localDimensions.nx);
            maxNy = Math.max(maxNy, chunk.localDimensions.ny);
            maxNz = Math.max(maxNz, chunk.localDimensions.nz);
        }

        const ghosts = dataContract.descriptor.requirements.ghostCells || 0;
        const pNx = maxNx + 2 * ghosts;
        const pNy = maxNy + 2 * ghosts;
        const pNz = maxNz > 1 ? maxNz + 2 * ghosts : 1;
        this.strideRow = pNx;
        this.cellsPerFaceRaw = pNx * pNy * pNz;

        const bytesPerFaceRaw = this.cellsPerFaceRaw * 4;
        const bytesPerFaceAligned = Math.ceil(bytesPerFaceRaw / 256) * 256;
        this.strideFace = bytesPerFaceAligned / 4;

        // 1. Calculate Field-based data (The Grid)
        this.totalStandardSlotsPerChunk = this.faceMappings
            .filter(f => !(f.type || 'scalar').startsWith('atomic'))
            .reduce((acc, f) => acc + (f.numComponents * f.numSlots), 0);

        this.totalAtomicSlotsPerChunk = this.faceMappings
            .filter(f => (f.type || 'scalar').startsWith('atomic'))
            .reduce((acc, f) => acc + (f.numComponents * f.numSlots), 0);

        this.standardByteLength = this.vGrid.chunks.length * this.totalStandardSlotsPerChunk * bytesPerFaceAligned;
        this.atomicByteLength = this.vGrid.chunks.length * this.totalAtomicSlotsPerChunk * bytesPerFaceAligned;

        // 2. Add Global Workspace (Scalars/Reductions - NOT scaled by Grid resolution)
        // Direct-Read Manifest Architecture (v9): Globals are now completely isolated
        // in a dedicated GPU buffer and no longer pollute the host grid.
        
        this.byteLength = this.standardByteLength + this.atomicByteLength;
    }

    public getBinding(faceIdx: number): 0 | 2 {
        const type = this.faceMappings[faceIdx].type || 'scalar';
        return type.startsWith('atomic') ? 2 : 0;
    }

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

    public getGlobalOffset(name: string): number {
        const gm = this.globalMappings.find(g => g.name === name);
        if (!gm) throw new Error(`Global variable '${name}' not found.`);
        
        // Direct-read architecture (v9): Globals have a dedicated buffer
        // Provide the offset *relative to the global buffer*
        return gm.byteOffset / 4;
    }

    public getLogicalFaceOffset(chunkIdx: number, faceIdx: number, slotIdx: number = 0): number {
        const binding = this.getBinding(faceIdx);
        const bufferOffset = (binding === 0) ? 0 : this.standardByteLength / 4;
        return bufferOffset + this.getFaceOffset(chunkIdx, faceIdx, slotIdx);
    }

    /** @deprecated - Use totalStandardSlotsPerChunk or totalAtomicSlotsPerChunk */
    public get totalSlotsPerChunk(): number {
        return this.totalStandardSlotsPerChunk + this.totalAtomicSlotsPerChunk;
    }
}

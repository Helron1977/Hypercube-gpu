import { IVirtualGrid, VirtualChunk } from '../topology/GridAbstractions';
import { MasterBuffer } from '../memory/MasterBuffer';
import { ParityManager } from '../ParityManager';
import { HypercubeGPUContext } from '../gpu/HypercubeGPUContext';
import { TopologyResolver, ResolvedTopology } from '../topology/TopologyResolver';
import { UniformLayoutValues } from './UniformLayout';

interface ChunkDispatchMetadata {
    vChunk: VirtualChunk;
    uniformOffset: number;
    dataOffset: number;
    dataSize: number;
    globalIdx: number;
    topology: ResolvedTopology;
}

/**
 * Manages Uniform buffers and staging for GpuDispatcher.
 * v6.0: Optimized for decoupled Standard/Atomic layouts and Binding 3 support.
 */
export class UniformStagingManager {
    public uniformBuffer?: GPUBuffer;
    public stagingBuffer?: Uint32Array;
    private dispatchMetadata: ChunkDispatchMetadata[] = [];
    private bytesPerChunkAligned: number = 0;
    private topologyResolver: TopologyResolver = new TopologyResolver();
    public paramNames: string[] = [];

    constructor(
        private vGrid: IVirtualGrid,
        private buffer: MasterBuffer,
        private parityManager: ParityManager
    ) {
        this.bytesPerChunkAligned = HypercubeGPUContext.alignToUniform(512);
        
        // Extract parameter order for semantic mapping
        if (this.vGrid.config && this.vGrid.config.params) {
            this.paramNames = Object.keys(this.vGrid.config.params);
        }

        this.refreshMetadata();
    }

    public refreshMetadata(): void {
        const strideFace = this.buffer.strideFace;
        // v6.0 FIX: Standard buffer size must ONLY account for Standard slots.
        const standardSlotsPerChunk = this.buffer.layout.totalStandardSlotsPerChunk;
        const maxRules = this.vGrid.config?.maxRules || 8;

        this.dispatchMetadata = this.vGrid.chunks.map((vChunk: VirtualChunk, i: number) => {
            const gid = vChunk.y * this.vGrid.chunkLayout.x + vChunk.x;
            return {
                vChunk,
                uniformOffset: i * maxRules * this.bytesPerChunkAligned,
                dataOffset: gid * standardSlotsPerChunk * strideFace * 4,
                dataSize: standardSlotsPerChunk * strideFace * 4,
                globalIdx: gid,
                topology: this.topologyResolver.resolve(vChunk, this.vGrid.chunkLayout, this.vGrid.config.boundaries)
            };
        });

        const totalUniformSize = this.vGrid.chunks.length * maxRules * this.bytesPerChunkAligned;
        this.stagingBuffer = new Uint32Array(totalUniformSize / 4);
    }

    public getMetadata() {
        return this.dispatchMetadata;
    }

    public ensureBuffers(device: GPUDevice) {
        const maxRules = this.vGrid.config?.maxRules || 8;
        const totalUniformSize = this.vGrid.chunks.length * maxRules * this.bytesPerChunkAligned;
        if (!this.uniformBuffer || this.uniformBuffer.size < totalUniformSize) {
            if (this.uniformBuffer) this.uniformBuffer.destroy();
            this.uniformBuffer = device.createBuffer({
                size: totalUniformSize,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                label: 'GpuDispatcher Uniforms (Modular)'
            });
        }
    }

    public updateStaging(t: number, rules: any[], faceMappings: any[], overrides?: Record<string, number>): void {
        const u32Data = this.stagingBuffer!;
        const f32Data = new Float32Array(u32Data.buffer);
        const maxRules = this.vGrid.config?.maxRules || 8;
        const descriptor = this.vGrid.dataContract.descriptor;
        const strideFace = this.buffer.strideFace;

        for (let rIdx = 0; rIdx < rules.length; rIdx++) {
            const scheme = rules[rIdx];
            for (let i = 0; i < this.dispatchMetadata.length; i++) {
                const meta = this.dispatchMetadata[i];
                const base = (i * maxRules + rIdx) * (this.bytesPerChunkAligned / 4);
                const dim = meta.vChunk.localDimensions;
                const ghosts = descriptor.requirements.ghostCells || 0;

                const layoutStrideRow = this.buffer.layout?.strideRow || (dim.nx + (ghosts * 2));
                const layoutPhysicalNy = dim.ny + (ghosts * 2);

                u32Data[base + UniformLayoutValues.NX] = dim.nx;
                u32Data[base + UniformLayoutValues.NY] = dim.ny;
                u32Data[base + UniformLayoutValues.STRIDE_ROW] = layoutStrideRow;
                u32Data[base + UniformLayoutValues.PHYSICAL_NY] = Math.floor(layoutPhysicalNy);

                f32Data[base + UniformLayoutValues.T] = t;
                u32Data[base + UniformLayoutValues.TICK] = this.parityManager.currentTick;
                u32Data[base + UniformLayoutValues.STRIDE_FACE] = strideFace;
                u32Data[base + UniformLayoutValues.NUM_FACES] = faceMappings.length;

                // Parameters mapping
                const configParams = this.vGrid.config.params || {};
                for (let p = 0; p < 8; p++) {
                    const genericKey = `p${p}`;
                    const semanticKey = this.paramNames[p];
                    let val = 0;
                    if (overrides && semanticKey && overrides[semanticKey] !== undefined) val = overrides[semanticKey];
                    else if (overrides && overrides[genericKey] !== undefined) val = overrides[genericKey];
                    else if (scheme.params) {
                        if (semanticKey && scheme.params[semanticKey] !== undefined) val = scheme.params[semanticKey] as number;
                        else if (scheme.params[genericKey] !== undefined) val = scheme.params[genericKey] as number;
                        else if (semanticKey) val = configParams[semanticKey];
                    } else if (semanticKey) val = configParams[semanticKey];
                    f32Data[base + UniformLayoutValues.PARAMS_START + p] = (typeof val === 'number') ? val : 0;
                }

                // Field Face Mapping (Slots 0-63)
                for (let f = 0; f < 64; f++) u32Data[base + UniformLayoutValues.FACES_START + f] = 0;

                for (let fIdx = 0; fIdx < Math.min(faceMappings.length, 64); fIdx++) {
                    const fm = faceMappings[fIdx];
                    const r = this.parityManager.getFaceIndices(fm.name);
                    const slotBase = UniformLayoutValues.FACES_START + fm.pointerOffset;
                    const slots = [r.read, r.write, r.old, r.base];
                    for (let s = 0; s < Math.min(fm.numSlots, 4); s++) {
                        u32Data[base + slotBase + s] = slots[s];
                    }
                }

                // Boundaries & Ghosts
                const topo = meta.topology;
                u32Data[base + UniformLayoutValues.GHOSTS] = ghosts;
                u32Data[base + UniformLayoutValues.LEFT_ROLE] = topo.leftRole;
                u32Data[base + UniformLayoutValues.RIGHT_ROLE] = topo.rightRole;
                u32Data[base + UniformLayoutValues.TOP_ROLE] = topo.topRole;
                u32Data[base + UniformLayoutValues.BOTTOM_ROLE] = topo.bottomRole;
                u32Data[base + UniformLayoutValues.FRONT_ROLE] = topo.frontRole;
                u32Data[base + UniformLayoutValues.BACK_ROLE] = topo.backRole;
            }
        }
    }

    public getBytesPerChunkAligned(): number {
        return this.bytesPerChunkAligned;
    }

    public destroy(): void {
        if (this.uniformBuffer) this.uniformBuffer.destroy();
    }
}

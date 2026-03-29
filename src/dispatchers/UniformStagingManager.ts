import { IVirtualGrid, VirtualChunk } from '../topology/GridAbstractions';
import { MasterBuffer } from '../memory/MasterBuffer';
import { ParityManager } from '../ParityManager';
import { HypercubeGPUContext } from '../gpu/HypercubeGPUContext';
import { TopologyResolver, ResolvedTopology } from '../topology/TopologyResolver';

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
 */
export class UniformStagingManager {
    public uniformBuffer?: GPUBuffer;
    public stagingBuffer?: Uint32Array;
    private dispatchMetadata: ChunkDispatchMetadata[] = [];
    private bytesPerChunkAligned: number = 0;
    private topologyResolver: TopologyResolver = new TopologyResolver();

    constructor(
        private vGrid: IVirtualGrid,
        private buffer: MasterBuffer,
        private parityManager: ParityManager
    ) {
        this.bytesPerChunkAligned = HypercubeGPUContext.alignToUniform(512);
        this.refreshMetadata();
    }

    public refreshMetadata(): void {
        const strideFace = this.buffer.strideFace;
        const totalSlots = this.buffer.totalSlotsPerChunk;
        const maxRules = this.vGrid.config?.maxRules || 8;

        this.dispatchMetadata = this.vGrid.chunks.map((vChunk, i) => {
            const gid = vChunk.y * this.vGrid.chunkLayout.x + vChunk.x;
            return {
                vChunk,
                uniformOffset: i * maxRules * this.bytesPerChunkAligned,
                dataOffset: gid * totalSlots * strideFace * 4,
                dataSize: totalSlots * strideFace * 4,
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

                u32Data[base + 0] = dim.nx;
                u32Data[base + 1] = dim.ny;
                u32Data[base + 2] = layoutStrideRow;
                u32Data[base + 3] = Math.floor(layoutPhysicalNy);

                f32Data[base + 4] = t;
                u32Data[base + 5] = this.parityManager.currentTick;
                u32Data[base + 6] = strideFace;
                u32Data[base + 7] = faceMappings.length;

                const configParams = this.vGrid.config.params || {};
                for (let p = 0; p < 8; p++) {
                    const key = `p${p}`;
                    let val = (overrides && overrides[key] !== undefined) 
                        ? overrides[key] 
                        : (scheme.params && scheme.params[key] !== undefined) ? scheme.params[key] : configParams[key];
                    f32Data[base + 8 + p] = (typeof val === 'number') ? val : 0;
                }

                const explicitFaces = scheme.faces || [];
                for (let f = 0; f < 64; f++) {
                    let faceNameWithSuffix = "";
                    if (f < explicitFaces.length) {
                        faceNameWithSuffix = explicitFaces[f];
                    } else if (f < faceMappings.length && explicitFaces.length === 0) {
                        faceNameWithSuffix = faceMappings[f].name;
                    }

                    if (faceNameWithSuffix) {
                        const faceName = faceNameWithSuffix.replace('.read', '').replace('.write', '').replace('.now', '').replace('.old', '').replace('.next', '');
                        const res = this.parityManager.getFaceIndices(faceName);

                        if (faceNameWithSuffix.endsWith('.read') || faceNameWithSuffix.endsWith('.now')) {
                            u32Data[base + 16 + f] = res.read;
                        } else if (faceNameWithSuffix.endsWith('.write') || faceNameWithSuffix.endsWith('.next')) {
                            u32Data[base + 16 + f] = res.write;
                        } else if (faceNameWithSuffix.endsWith('.old')) {
                            u32Data[base + 16 + f] = res.old;
                        } else {
                            u32Data[base + 16 + f] = res.base;
                        }
                    } else {
                        u32Data[base + 16 + f] = 0;
                    }
                }

                // Boundaries
                const topo = meta.topology;
                u32Data[base + 80] = ghosts;
                u32Data[base + 81] = topo.leftRole;
                u32Data[base + 82] = topo.rightRole;
                u32Data[base + 83] = topo.topRole;
                u32Data[base + 84] = topo.bottomRole;
                u32Data[base + 85] = topo.frontRole;
                u32Data[base + 86] = topo.backRole;
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

import { IMasterBuffer, IPhysicalChunk, IVirtualGrid } from '../topology/GridAbstractions';
import { DataContract } from '../DataContract';
import { ParityManager } from '../ParityManager';
import { HypercubeGPUContext } from '../gpu/HypercubeGPUContext';
import { MemoryLayout } from './MemoryLayout';

/**
 * The MasterBuffer is the physical memory anchor for Hypercube GPU Core.
 * v6.0 SOTA: Optimized for Dedicated Global Workspace (Binding 3).
 */
export class MasterBuffer implements IMasterBuffer {
    public readonly rawBuffer: ArrayBuffer;
    public readonly gpuBuffer: GPUBuffer;
    public readonly gpuAtomicBuffer?: GPUBuffer;
    public readonly gpuGlobalBuffer?: GPUBuffer;
    public readonly gpuGlobalStagingBuffer?: GPUBuffer;
    public readonly layout: MemoryLayout;

    private chunkViews: Map<string, IPhysicalChunk> = new Map();

    public readonly parityManager: ParityManager;
    private device: GPUDevice;
    private vGrid: IVirtualGrid;

    public readonly byteOffset: number = 0;
    public readonly atomicOffset: number = 0;
    public readonly globalOffset: number = 0;

    constructor(
        vGrid: IVirtualGrid, 
        parityManager?: ParityManager, 
        mockDevice?: any,
        sharedGpuBuffer?: GPUBuffer,
        sharedGpuAtomicBuffer?: GPUBuffer,
        sharedGpuGlobalBuffer?: GPUBuffer,
        byteOffset = 0,
        atomicOffset = 0,
        globalOffset = 0
    ) {
        this.vGrid = vGrid;
        this.parityManager = parityManager || new ParityManager(vGrid.dataContract);
        this.layout = new MemoryLayout(this.vGrid);
        this.device = mockDevice || HypercubeGPUContext.device;

        this.byteOffset = byteOffset;
        this.atomicOffset = atomicOffset;
        this.globalOffset = globalOffset;

        // 1. Standard Buffer (Binding 0)
        if (sharedGpuBuffer) {
            this.gpuBuffer = sharedGpuBuffer;
        } else {
            this.gpuBuffer = this.device.createBuffer({
                size: Math.ceil(this.layout.standardByteLength / 4) * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
                label: 'Hypercube Master Buffer (Float32)'
            });
        }

        // 2. Atomic Buffer (Binding 2)
        if (this.layout.atomicByteLength > 0) {
            if (sharedGpuAtomicBuffer) {
                this.gpuAtomicBuffer = sharedGpuAtomicBuffer;
            } else {
                this.gpuAtomicBuffer = this.device.createBuffer({
                    size: Math.ceil(this.layout.atomicByteLength / 4) * 4,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
                    label: 'Hypercube Atomic Buffer (U32)'
                });
            }
        }

        // 3. Global Buffer (Binding 3)
        const globalBytes = this.vGrid.dataContract.calculateGlobalBytes();
        if (globalBytes > 0) {
            if (sharedGpuGlobalBuffer) {
                this.gpuGlobalBuffer = sharedGpuGlobalBuffer;
            } else {
                this.gpuGlobalBuffer = this.device.createBuffer({
                    size: Math.max(globalBytes, 16),
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
                    label: 'Hypercube Global Workspace (Direct-Read)'
                });
            }
            this.gpuGlobalStagingBuffer = this.device.createBuffer({
                size: Math.max(globalBytes, 16),
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
                label: 'Hypercube Global Workspace (Staging)'
            });
        }

        // 4. Host-side Memory Partitioning
        this.rawBuffer = new ArrayBuffer(this.layout.byteLength);
        this.partitionMemory();
    }

    public get byteLength(): number { return this.layout.byteLength; }
    public get strideFace(): number { return this.layout.strideFace; }
    public get totalSlotsPerChunk(): number { return this.layout.totalSlotsPerChunk; }

    private partitionMemory() {
        // Face Partitioning
        for (let i = 0; i < this.vGrid.chunks.length; i++) {
            const vChunk = this.vGrid.chunks[i];
            const physicalFaces: Float32Array[] = [];
            for (let fIdx = 0; fIdx < this.layout.faceMappings.length; fIdx++) {
                const face = this.layout.faceMappings[fIdx];
                for (let b = 0; b < face.numSlots; b++) {
                    const offset = this.layout.getLogicalFaceOffset(i, fIdx, b);
                    const viewLength = face.numComponents * this.strideFace;
                    physicalFaces.push(new Float32Array(this.rawBuffer, offset * 4, viewLength));
                }
            }
            this.chunkViews.set(vChunk.id, { id: vChunk.id, faces: physicalFaces });
        }
    }

    /**
     * Synchronizes specific faces from GPU to Host.
     */
    public async syncFacesToHost(faces: (string | number)[]): Promise<void> {
        const faceMappings = this.layout.faceMappings;
        const faceIndices = faces.map(f => typeof f === 'string' ? faceMappings.findIndex(fm => fm.name === f) : f);

        const encoder = this.device.createCommandEncoder();
        const staging: { buffer: GPUBuffer, cpuOffset: number, bytes: number }[] = [];

        faceIndices.forEach(fIdx => {
            if (fIdx === -1) return;
            const mapping = faceMappings[fIdx];
            const binding = this.layout.getBinding(fIdx);
            const src = (binding === 0) ? this.gpuBuffer : this.gpuAtomicBuffer;
            if (!src) return;
            const baseOffset = (binding === 0) ? this.byteOffset : this.atomicOffset;

            for (let c = 0; c < this.vGrid.chunks.length; c++) {
                for (let s = 0; s < mapping.numSlots; s++) {
                    const bytes = mapping.numComponents * this.strideFace * 4;
                    const gpuOff = baseOffset + this.layout.getFaceOffset(c, fIdx, s) * 4;
                    const cpuOff = this.layout.getLogicalFaceOffset(c, fIdx, s) * 4;

                    const stg = this.device.createBuffer({
                        size: bytes,
                        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
                        label: `Face Sync Staging (fIdx:${fIdx}, sIdx:${s})`
                    });
                    encoder.copyBufferToBuffer(src, gpuOff, stg, 0, bytes);
                    staging.push({ buffer: stg, cpuOffset: cpuOff, bytes });
                }
            }
        });

        this.device.queue.submit([encoder.finish()]);
        await Promise.all(staging.map(s => s.buffer.mapAsync(GPUMapMode.READ)));

        staging.forEach(s => {
            new Uint8Array(this.rawBuffer, s.cpuOffset, s.bytes).set(new Uint8Array(s.buffer.getMappedRange()));
            s.buffer.unmap();
            s.buffer.destroy();
        });
    }

    /**
     * Direct-Read Manifest Architecture (v9).
     * Reads a specific global variable on-demand without touching host grid data.
     */
    public async readGlobalData(name: string): Promise<Float32Array | Int32Array> {
        if (!this.gpuGlobalBuffer || !this.gpuGlobalStagingBuffer) throw new Error("No global buffer allocated.");
        const globals = this.vGrid.dataContract.getGlobalMappings();
        const gm = globals.find(g => g.name === name);
        if (!gm) throw new Error(`Global Context '${name}' not found.`);

        const isAtomicI32 = gm.type === 'atomic_i32' || gm.type === 'atomic';
        const byteSize = gm.count * 4;
        const byteOffset = gm.byteOffset;

        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.gpuGlobalBuffer, this.globalOffset + byteOffset, this.gpuGlobalStagingBuffer, byteOffset, byteSize);
        this.device.queue.submit([encoder.finish()]);

        await this.gpuGlobalStagingBuffer.mapAsync(GPUMapMode.READ, byteOffset, byteSize);
        const arrayBuffer = this.gpuGlobalStagingBuffer.getMappedRange(byteOffset, byteSize);
        const localCopy = isAtomicI32 ? new Int32Array(arrayBuffer.slice(0)) : new Float32Array(arrayBuffer.slice(0));
        this.gpuGlobalStagingBuffer.unmap();

        return localCopy;
    }

    public async syncToHost(): Promise<void> {
        await this.syncFacesToHost(this.layout.faceMappings.map(f => f.name));
    }

    public syncToDevice(): void {
        this.device.queue.writeBuffer(this.gpuBuffer, this.byteOffset, new Uint8Array(this.rawBuffer, 0, this.layout.standardByteLength));
        if (this.gpuAtomicBuffer) {
            this.device.queue.writeBuffer(this.gpuAtomicBuffer, this.atomicOffset, new Uint8Array(this.rawBuffer, this.layout.standardByteLength, this.layout.atomicByteLength));
        }
    }

    public getChunkViews(chunkId: string): IPhysicalChunk {
        const v = this.chunkViews.get(chunkId);
        if (!v) throw new Error(`Chunk ${chunkId} not partitioned.`);
        return v;
    }

    public setFaceData(chunkId: string, faceName: string, data: Float32Array | number[], fillAll = false, parity?: number): void {
        const views = this.getChunkViews(chunkId);
        const idx = this.layout.faceMappings.findIndex(f => f.name === faceName);
        if (idx === -1) return;
        const m = this.layout.faceMappings[idx];
        const res = this.parityManager.getFaceIndices(faceName);
        let base = 0; for (let i = 0; i < idx; i++) base += this.layout.faceMappings[i].numSlots;
        const target = (parity !== undefined) ? (parity % m.numSlots) : res.slotRead;
        views.faces[base + target].set(data as Float32Array);
        if (fillAll && m.numSlots > 1) {
            for (let s = 0; s < m.numSlots; s++) views.faces[base + s].set(data as Float32Array);
        }
    }

    public getFaceData(chunkId: string, faceName: string, parity?: number): Float32Array {
        const views = this.getChunkViews(chunkId);
        const idx = this.layout.faceMappings.findIndex(f => f.name === faceName);
        if (idx === -1) throw new Error(`Face ${faceName} not found.`);
        const m = this.layout.faceMappings[idx];
        const res = this.parityManager.getFaceIndices(faceName);
        let base = 0; for (let i = 0; i < idx; i++) base += this.layout.faceMappings[i].numSlots;
        const target = (parity !== undefined) ? (parity % m.numSlots) : res.slotRead;
        return views.faces[base + target];
    }

    public destroy(): void {
        // Only destroy buffers if we own them (not shared)
        // If offsets are 0 and they weren't passed in, we own them.
        // Actually, it's safer to have explicit ownership flags, but let's check if they were passed in via Constructor logic if possible.
        // For now, if anyone calls destroy() on a linked simulation, it might be dangerous.
        // Standard practice: Shared buffers are destroyed by their manager.
    }
}

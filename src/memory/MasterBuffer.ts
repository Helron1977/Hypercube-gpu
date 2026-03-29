import { IMasterBuffer, IPhysicalChunk, IVirtualGrid } from '../topology/GridAbstractions';
import { DataContract } from '../DataContract';
import { HypercubeGPUContext } from '../gpu/HypercubeGPUContext';
import { MemoryLayout } from './MemoryLayout';

/**
 * The MasterBuffer is the physical memory anchor for Hypercube GPU Core.
 * It allocates a single contiguous WebGPU buffer and manages CPU/GPU sync.
 */
export class MasterBuffer implements IMasterBuffer {
    public readonly rawBuffer: ArrayBuffer;
    public readonly gpuBuffer: GPUBuffer;
    public readonly gpuAtomicBuffer?: GPUBuffer;
    public readonly layout: MemoryLayout;
    private chunkViews: Map<string, IPhysicalChunk> = new Map();

    private device: GPUDevice;
    private vGrid: IVirtualGrid;

    constructor(vGrid: IVirtualGrid, mockDevice?: any) {
        this.vGrid = vGrid;
        this.layout = new MemoryLayout(this.vGrid);

        if (mockDevice) {
            this.device = mockDevice;
        } else {
            if (!HypercubeGPUContext.isInitialized) {
                throw new Error("MasterBuffer: GPU Context not initialized.");
            }
            this.device = HypercubeGPUContext.device;
        }
        
        // 1. Allouer le MasterBuffer Standard (Binding 0)
        this.gpuBuffer = this.device.createBuffer({
            size: Math.ceil(this.layout.standardByteLength / 4) * 4,
            usage: 0x0080 | 0x0008 | 0x0004, // STORAGE | COPY_DST | COPY_SRC
            label: 'Hypercube Master Buffer (Float32)'
        });

        // 2. Allouer le MasterBuffer Atomique SI nécessaire (Binding 2)
        if (this.layout.atomicByteLength > 0) {
            this.gpuAtomicBuffer = this.device.createBuffer({
                size: Math.ceil(this.layout.atomicByteLength / 4) * 4,
                usage: 0x0080 | 0x0008 | 0x0004, // STORAGE | COPY_DST | COPY_SRC
                label: 'Hypercube Atomic Buffer (U32)'
            });
        }

        this.rawBuffer = new ArrayBuffer(this.layout.byteLength);
        this.partitionMemory();
    }

    // Compatibility getters
    public get byteLength(): number { return this.layout.byteLength; }
    public get strideFace(): number { return this.layout.strideFace; }
    public get totalSlotsPerChunk(): number { return this.layout.totalSlotsPerChunk; }

    private partitionMemory() {
        const grid = this.vGrid;

        for (let i = 0; i < grid.chunks.length; i++) {
            const vChunk = grid.chunks[i];
            const physicalFaces: Float32Array[] = [];

            for (let fIdx = 0; fIdx < this.layout.faceMappings.length; fIdx++) {
                const face = this.layout.faceMappings[fIdx];
                const numBuffers = face.numSlots;
                
                for (let b = 0; b < numBuffers; b++) {
                    const offset = this.layout.getLogicalFaceOffset(i, fIdx, b);
                    const viewLength = face.numComponents * this.strideFace;
                    
                    const view = new Float32Array(this.rawBuffer, offset * 4, viewLength);
                    physicalFaces.push(view);
                }
            }

            this.chunkViews.set(vChunk.id, {
                id: vChunk.id,
                faces: physicalFaces
            });
        }
    }

    public async syncToHost(): Promise<void> {
        // En v5.0.4, on synchronise les deux si présents
        await this.syncFacesToHost(this.layout.faceMappings.map(f => f.name));
    }

    public async syncFacesToHost(faces: (string | number)[]): Promise<void> {
        const dataContract = this.vGrid.dataContract;
        const faceIndices = faces.map(f => {
            if (typeof f === 'string') {
                const idx = dataContract.descriptor.faces.findIndex(face => face.name === f);
                if (idx === -1) throw new Error(`Face ${f} not found.`);
                return idx;
            }
            return f;
        });

        const copyTasks: { source: GPUBuffer, gpuOffset: number, cpuOffset: number, bytes: number }[] = [];
        
        faceIndices.forEach((fIdx) => {
            const faceMapping = dataContract.getFaceMappings()[fIdx];
            const numComponents = faceMapping.numComponents;
            const numBuffers = faceMapping.numSlots;
            const binding = this.layout.getBinding(fIdx);
            const sourceBuffer = (binding === 0) ? this.gpuBuffer : this.gpuAtomicBuffer;

            if (!sourceBuffer) return;

            for (let chunkIdx = 0; chunkIdx < this.vGrid.chunks.length; chunkIdx++) {
                for (let b = 0; b < numBuffers; b++) {
                    const bytes = numComponents * this.strideFace * 4;
                    const gpuOffset = this.layout.getFaceOffset(chunkIdx, fIdx, b) * 4;
                    const cpuOffset = this.layout.getLogicalFaceOffset(chunkIdx, fIdx, b) * 4;
                    
                    copyTasks.push({ source: sourceBuffer, gpuOffset, cpuOffset, bytes });
                }
            }
        });

        const stagingBuffers = copyTasks.map(t => this.device.createBuffer({
            size: t.bytes,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        }));

        const encoder = this.device.createCommandEncoder();
        copyTasks.forEach((task, i) => {
            encoder.copyBufferToBuffer(task.source, task.gpuOffset, stagingBuffers[i], 0, task.bytes);
        });

        this.device.queue.submit([encoder.finish()]);
        await Promise.all(stagingBuffers.map(b => b.mapAsync(GPUMapMode.READ)));

        copyTasks.forEach((task, i) => {
            const data = new Uint8Array(stagingBuffers[i].getMappedRange());
            const cpuView = new Uint8Array(this.rawBuffer, task.cpuOffset, task.bytes);
            cpuView.set(data);
            stagingBuffers[i].unmap();
            stagingBuffers[i].destroy();
        });
    }

    public destroy(): void {
        if (this.gpuBuffer) this.gpuBuffer.destroy();
        if (this.gpuAtomicBuffer) this.gpuAtomicBuffer.destroy();
    }

    public syncToDevice(): void {
        // 1. Envoyer le buffer Standard
        this.device.queue.writeBuffer(
            this.gpuBuffer, 0, 
            new Uint8Array(this.rawBuffer, 0, this.layout.standardByteLength)
        );

        // 2. Envoyer le buffer Atomique si présent
        if (this.gpuAtomicBuffer) {
            this.device.queue.writeBuffer(
                this.gpuAtomicBuffer, 0, 
                new Uint8Array(this.rawBuffer, this.layout.standardByteLength, this.layout.atomicByteLength)
            );
        }
    }

    public getChunkViews(chunkId: string): IPhysicalChunk {
        const views = this.chunkViews.get(chunkId);
        if (!views) throw new Error(`Chunk ${chunkId} not partitioned.`);
        return views;
    }

    public setFaceData(chunkId: string, faceName: string, data: Float32Array | number[], fillAllPingPong: boolean = false, parity?: number): void {
        const views = this.getChunkViews(chunkId);
        const dataContract = this.vGrid.dataContract;
        const faceIdx = dataContract.descriptor.faces.findIndex(f => f.name === faceName);

        if (faceIdx === -1) return;

        const m = this.layout.faceMappings[faceIdx];
        let bufIdx = 0;
        for (let i = 0; i < faceIdx; i++) {
            bufIdx += this.layout.faceMappings[i].numSlots;
        }

        const targetBufIdx = (parity !== undefined && m.numSlots > 1) 
                            ? bufIdx + (parity % m.numSlots)
                            : bufIdx;

        views.faces[targetBufIdx].set(data as Float32Array);
        if (fillAllPingPong && m.numSlots > 1) {
            for (let slot = 0; slot < m.numSlots; slot++) {
                views.faces[bufIdx + slot].set(data as Float32Array);
            }
        }
    }

    public getFaceData(chunkId: string, faceName: string, parity?: number): Float32Array {
        const views = this.getChunkViews(chunkId);
        const dataContract = this.vGrid.dataContract;
        const faceIdx = dataContract.descriptor.faces.findIndex(f => f.name === faceName);
        if (faceIdx === -1) throw new Error(`Face ${faceName} not found.`);

        const m = this.layout.faceMappings[faceIdx];
        let bufIdx = 0;
        for (let i = 0; i < faceIdx; i++) {
            bufIdx += this.layout.faceMappings[i].numSlots;
        }

        const targetBufIdx = (parity !== undefined && m.numSlots > 1) 
                            ? bufIdx + (parity % m.numSlots)
                            : bufIdx;

        return views.faces[targetBufIdx];
    }
}

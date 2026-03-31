import { HypercubeGPUContext } from '../gpu/HypercubeGPUContext';

/**
 * SharedMasterBuffer manages a single large GPUBuffer across multiple engines.
 * Enforces WebGPU 256-byte alignment requirements for sub-allocation.
 */
export class SharedMasterBuffer {
    public readonly gpuBuffer: GPUBuffer;
    public readonly gpuAtomicBuffer?: GPUBuffer;
    public readonly gpuGlobalBuffer?: GPUBuffer;
    
    private cursor: number = 0;
    private atomicCursor: number = 0;
    private globalCursor: number = 0;

    constructor(sizeBytes: number, atomicSizeBytes: number = 0, globalSizeBytes: number = 0) {
        if (!HypercubeGPUContext.isInitialized) {
            throw new Error("Hypercube GPU Context must be initialized before creating a SharedMasterBuffer.");
        }

        const device = HypercubeGPUContext.device;

        this.gpuBuffer = device.createBuffer({
            size: Math.ceil(sizeBytes / 256) * 256,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            label: 'Hypercube Shared Master Buffer (Standard)'
        });

        if (atomicSizeBytes > 0) {
            this.gpuAtomicBuffer = device.createBuffer({
                size: Math.ceil(atomicSizeBytes / 256) * 256,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
                label: 'Hypercube Shared Master Buffer (Atomic)'
            });
        }

        if (globalSizeBytes > 0) {
            this.gpuGlobalBuffer = device.createBuffer({
                size: Math.ceil(globalSizeBytes / 256) * 256,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
                label: 'Hypercube Shared Master Buffer (Global)'
            });
        }
    }

    /**
     * Allocates a slice from the shared buffer and returns its offset.
     */
    public allocate(bytes: number, type: 'standard' | 'atomic' | 'global' = 'standard'): number {
        let currentOffset = 0;
        const alignedBytes = Math.ceil(bytes / 256) * 256;

        if (type === 'standard') {
            currentOffset = this.cursor;
            this.cursor += alignedBytes;
            if (this.cursor > this.gpuBuffer.size) throw new Error("Out of memory on Shared standard buffer.");
        } else if (type === 'atomic') {
            if (!this.gpuAtomicBuffer) throw new Error("Atomic buffer not initialized in SharedMasterBuffer.");
            currentOffset = this.atomicCursor;
            this.atomicCursor += alignedBytes;
            if (this.atomicCursor > this.gpuAtomicBuffer.size) throw new Error("Out of memory on Shared atomic buffer.");
        } else {
            if (!this.gpuGlobalBuffer) throw new Error("Global buffer not initialized in SharedMasterBuffer.");
            currentOffset = this.globalCursor;
            this.globalCursor += alignedBytes;
            if (this.globalCursor > this.gpuGlobalBuffer.size) throw new Error("Out of memory on Shared global buffer.");
        }

        return currentOffset;
    }

    public destroy(): void {
        this.gpuBuffer.destroy();
        this.gpuAtomicBuffer?.destroy();
        this.gpuGlobalBuffer?.destroy();
    }
}

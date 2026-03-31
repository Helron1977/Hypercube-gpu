import { IVirtualGrid } from '../topology/GridAbstractions';
import { MasterBuffer } from '../memory/MasterBuffer';
import { PipelineCache } from './PipelineCache';
import { WgslHeaderResult } from './WgslHeaderGenerator';

/**
 * Handles the Halo (Ghost Cells) exchange pass for multi-chunk simulations.
 */
export class HaloExchangePass {
    private haloTransferBuffer?: GPUBuffer;
    private bindGroupCache: Map<string, GPUBindGroup> = new Map();

    constructor() {}

    public async execute(
        device: GPUDevice,
        vGrid: IVirtualGrid,
        buffer: MasterBuffer,
        dispatchMetadata: any[],
        pipelineCache: PipelineCache,
        getWgslHeader: (type: string, source: string) => WgslHeaderResult,
        kernels: Record<string, string>,
        encoder: GPUCommandEncoder
    ): Promise<void> {
        const source = kernels['HaloExchange'];
        if (!source) return;

        const metadata = await pipelineCache.getPipeline(device, 'HaloExchange', source, getWgslHeader);
        const pipeline = metadata.pipeline;
        
        // Zero-Stall Pre-check: Only update if anything changed
        const transfers: any[] = [];
        const stride = buffer.strideFace;
        const numFaces = vGrid.dataContract.getFaceMappings().length;

        for (const meta of dispatchMetadata) {
            for (const joint of meta.vChunk.joints) {
                if (joint.role !== 'joint') continue;
                const neighbor = vGrid.chunks.find(c => c.id === joint.neighborId);
                if (!neighbor) continue;
                const neighborMeta = dispatchMetadata.find(m => m.vChunk.id === neighbor.id);
                if (!neighborMeta) continue;

                const dim = meta.vChunk.localDimensions;
                const lx = dim.nx + 2;
                const ly = dim.ny + 2;
                const lz = (dim.nz || 1) + 2;

                let axis = 0, srcPos = 0, dstPos = 0, size1 = 0, size2 = 0;
                switch (joint.face) {
                    case 'left': axis = 0; srcPos = 1; dstPos = dim.nx + 1; break;
                    case 'right': axis = 0; srcPos = dim.nx; dstPos = 0; break;
                    case 'bottom': axis = 1; srcPos = 1; dstPos = dim.ny + 1; break;
                    case 'top': axis = 1; srcPos = dim.ny; dstPos = 0; break;
                    case 'back': axis = 2; srcPos = 1; dstPos = (dim.nz || 1) + 1; break;
                    case 'front': axis = 2; srcPos = (dim.nz || 1); dstPos = 0; break;
                    default: continue;
                }
                if (axis === 0) { size1 = ly; size2 = lz; }
                else if (axis === 1) { size1 = lx; size2 = lz; }
                else { size1 = lx; size2 = ly; }

                transfers.push({
                    srcBase: neighborMeta.dataOffset / 4,
                    dstBase: meta.dataOffset / 4,
                    stride, lx, ly, lz, numFaces, axis,
                    srcPos: (joint.face === 'left' || joint.face === 'bottom' || joint.face === 'back') ?
                        (axis === 0 ? neighbor.localDimensions.nx : (axis === 1 ? neighbor.localDimensions.ny : neighbor.localDimensions.nz)) : 1,
                    dstPos, size1, size2
                });
            }
        }
        if (transfers.length === 0) return;

        const transferSize = 12 * 4;
        if (!this.haloTransferBuffer || this.haloTransferBuffer.size < transfers.length * transferSize) {
            if (this.haloTransferBuffer) this.haloTransferBuffer.destroy();
            this.haloTransferBuffer = device.createBuffer({
                size: transfers.length * transferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                label: 'Halo Exchange Transfers'
            });
            this.bindGroupCache.clear();
        }

        const rawTransfers = new Uint32Array(transfers.length * 12);
        for (let i = 0; i < transfers.length; i++) {
            const t = transfers[i];
            const b = i * 12;
            rawTransfers[b + 0] = t.srcBase; rawTransfers[b + 1] = t.dstBase; rawTransfers[b + 2] = t.stride; rawTransfers[b + 3] = t.lx;
            rawTransfers[b + 4] = t.ly; rawTransfers[b + 5] = t.lz; rawTransfers[b + 6] = t.numFaces; rawTransfers[b + 7] = t.axis;
            rawTransfers[b + 8] = t.srcPos; rawTransfers[b + 9] = t.dstPos; rawTransfers[b + 10] = t.size1; rawTransfers[b + 11] = t.size2;
        }
        device.queue.writeBuffer(this.haloTransferBuffer, 0, rawTransfers);

        const passEncoder = encoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        
        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: buffer.gpuBuffer } }
        ];

        // Binding 2: Field Atomics (needed if Halo Exchange uses atomic operations)
        if (metadata.usesAtomics && buffer.gpuAtomicBuffer) {
            entries.push({ binding: 2, resource: { buffer: buffer.gpuAtomicBuffer } });
        } else if (!metadata.usesAtomics) {
            // Standard Halo uses Binding 2 for its transfer instructions if not using dataAtomic
            const b = this.haloTransferBuffer;
            if (b) entries.push({ binding: 2, resource: { buffer: b } });
        }

        // Binding 3: Global Workspace (if required by Halo kernel)
        if (metadata.usesGlobals) {
            const globals = vGrid.dataContract.getGlobalMappings();
            if (globals.length > 0) {
                const hasAtomic = globals.some(g => g.type.startsWith('atomic'));
                const sourceBuffer = (hasAtomic ? buffer.gpuAtomicBuffer : buffer.gpuBuffer) as GPUBuffer;
                if (!sourceBuffer) return; // Should not happen in production
                const firstGlobalName = globals[0].name;
                const globalOffset = buffer.layout.getGlobalOffset(firstGlobalName) * 4;
                const globalSize = vGrid.dataContract.calculateGlobalBytes();
                
                entries.push({
                    binding: 3,
                    resource: { buffer: sourceBuffer, offset: globalOffset, size: Math.max(globalSize, 16) }
                });
            }
        }
        
        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries
        });
        
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(16, 16, transfers.length);
        passEncoder.end();
    }

    public destroy(): void {
        const b = this.haloTransferBuffer;
        if (b) b.destroy();
        this.bindGroupCache.clear();
    }
}

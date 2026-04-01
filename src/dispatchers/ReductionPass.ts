import { IVirtualGrid } from '../topology/GridAbstractions';
import { MasterBuffer } from '../memory/MasterBuffer';
import { ParityManager } from '../ParityManager';
import { PipelineCache } from './PipelineCache';
import { UniformStagingManager } from './UniformStagingManager';
import { UniformLayoutValues } from './UniformLayout';
import { WgslHeaderResult } from './WgslHeaderGenerator';

/**
 * Handles GPU Reduction passes (sum, min, max using atomics).
 */
export class ReductionPass {
    private forceResultsBuffer?: GPUBuffer;
    private forceResultsStaging?: GPUBuffer;

    constructor() {}

    /**
     * Performs a GPU reduction and returns the results on the host.
     * @param precisionScale Factor to convert floats to Int32 atomics (default 1e9 for high precision).
     */
    public async reduce(
        device: GPUDevice,
        vGrid: IVirtualGrid,
        buffer: MasterBuffer,
        parityManager: ParityManager,
        pipelineCache: PipelineCache,
        stagingManager: UniformStagingManager,
        getWgslHeader: (type: string, source: string) => WgslHeaderResult,
        source: string,
        faceName?: string,
        numResults: number = 1,
        precisionScale: number = 1000000000.0
    ): Promise<number[]> {
        if (!this.forceResultsBuffer) {
            this.forceResultsBuffer = device.createBuffer({
                size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
                label: 'Reduction Results (GPU)'
            });
            this.forceResultsStaging = device.createBuffer({
                size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
                label: 'Reduction Results (Staging)'
            });
        }
        
        device.queue.writeBuffer(this.forceResultsBuffer, 0, new Int32Array(4).fill(0));
        
        const metadata = await pipelineCache.getPipeline(device, 'Reduction', source, getWgslHeader, 'main');
        const pipeline = metadata.pipeline;
        const encoder = device.createCommandEncoder();
        
        stagingManager.ensureBuffers(device);
        const dispatchMetadata = stagingManager.getMetadata();
        const stagingBuffer = stagingManager.stagingBuffer!;
        const f32View = new Float32Array(stagingBuffer.buffer);
        const faceMappings = vGrid.dataContract.getFaceMappings();
        const maxRules = vGrid.config?.maxRules || 8;
        const bytesPerChunkAligned = stagingManager.getBytesPerChunkAligned();

        for (let i = 0; i < dispatchMetadata.length; i++) {
            const meta = dispatchMetadata[i];
            const base = (i * maxRules) * (bytesPerChunkAligned / 4);
            const dim = meta.vChunk.localDimensions;
            const ghosts = vGrid.dataContract.descriptor.requirements.ghostCells || 1;
            const layoutStrideRow = buffer.layout?.strideRow || (dim.nx + (ghosts * 2));
            const layoutPhysicalNy = dim.ny + (ghosts * 2);

            stagingBuffer[base + UniformLayoutValues.NX] = dim.nx;
            stagingBuffer[base + UniformLayoutValues.NY] = dim.ny;
            stagingBuffer[base + UniformLayoutValues.STRIDE_ROW] = layoutStrideRow;
            stagingBuffer[base + UniformLayoutValues.PHYSICAL_NY] = Math.floor(layoutPhysicalNy);
            stagingBuffer[base + UniformLayoutValues.TICK] = parityManager.currentTick;
            stagingBuffer[base + UniformLayoutValues.STRIDE_FACE] = buffer.strideFace;
            stagingBuffer[base + UniformLayoutValues.NUM_FACES] = faceMappings.length;
            stagingBuffer[base + UniformLayoutValues.GHOSTS] = ghosts;

            const configParams = vGrid.config.params || {};
            for (let p = 0; p < 8; p++) {
                const key = `p${p}`;
                if (configParams[key] !== undefined) f32View[base + UniformLayoutValues.PARAMS_START + p] = configParams[key];
            }

            // High Precision Scaling for Atomic-based Reduction (mapped to p7)
            f32View[base + UniformLayoutValues.PARAMS_START + 7] = precisionScale;

            // Clear the faces section with zeroes first
            for (let f = 0; f < 64; f++) stagingBuffer[base + UniformLayoutValues.FACES_START + f] = 0;

            // Elite Alignment: If a specific face is requested, target its 'Now' slot
            if (faceName) {
                const fm = faceMappings.find(m => m.name === faceName);
                if (fm) {
                    // Target the correct current read slot dynamically
                    f32View[base + UniformLayoutValues.PARAMS_START + 0] = parityManager.getFaceIndices(fm.name).read;
                }
            }

            for (let fIdx = 0; fIdx < Math.min(faceMappings.length, 64); fIdx++) {
                const fm = faceMappings[fIdx];
                const res = parityManager.getFaceIndices(fm.name);
                
                // Elite Automatic Mapping (Dynamic Slots: Base, Now, Next, Old)
                const slotBase = UniformLayoutValues.FACES_START + fm.pointerOffset;
                stagingBuffer[base + slotBase + 0] = res.base;
                if (fm.numSlots >= 4) {
                    stagingBuffer[base + slotBase + 1] = res.read;
                    stagingBuffer[base + slotBase + 2] = res.write;
                    stagingBuffer[base + slotBase + 3] = res.old;
                } else if (fm.numSlots >= 2) {
                    stagingBuffer[base + slotBase + 1] = res.read;
                }
            }
        }
        device.queue.writeBuffer(stagingManager.uniformBuffer!, 0, stagingBuffer.buffer);

        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        for (let i = 0; i < dispatchMetadata.length; i++) {
            const meta = dispatchMetadata[i];
            const ruleOffset = 0; 
            
            const entries: GPUBindGroupEntry[] = [
                { binding: 0, resource: { buffer: buffer.gpuBuffer, offset: meta.dataOffset, size: meta.dataSize } },
                { binding: 1, resource: { buffer: stagingManager.uniformBuffer!, offset: meta.uniformOffset + ruleOffset, size: bytesPerChunkAligned } }
            ];

            // Binding 2: Field Atomics OR Legacy Results Buffer
            if (metadata.usesAtomics) {
                const atomicBuffer = this.forceResultsBuffer || buffer.gpuAtomicBuffer;
                if (atomicBuffer) {
                    entries.push({ binding: 2, resource: { buffer: atomicBuffer } });
                }
            }

            // Binding 3: Global Workspace (v6.0 SOTA Reduction)
            if (metadata.usesGlobals) {
                const globals = vGrid.dataContract.getGlobalMappings();
                if (globals.length > 0) {
                    if (!buffer.gpuGlobalBuffer) {
                        throw new Error(`ReductionPass: Dedicated GPU Global Buffer not available.`);
                    }
                    const globalSize = vGrid.dataContract.calculateGlobalBytes();
                    entries.push({
                        binding: 3,
                        resource: { buffer: buffer.gpuGlobalBuffer, offset: 0, size: Math.max(globalSize, 16) }
                    });
                }
            }
            
            const bindGroup = device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries
            });
            
            pass.setBindGroup(0, bindGroup);
            const dim = meta.vChunk.localDimensions;
            const nz = dim.nz || 1;
            
            // Adjust workgroup count based on metadata (8,8,4 for 3D, 16,16,1 for 2D)
            if (nz > 1) {
                pass.dispatchWorkgroups(Math.ceil(dim.nx / 8), Math.ceil(dim.ny / 8), Math.ceil(nz / 4));
            } else {
                pass.dispatchWorkgroups(Math.ceil(dim.nx / 16), Math.ceil(dim.ny / 16), 1);
            }
        }
        pass.end();
        
        encoder.copyBufferToBuffer(this.forceResultsBuffer, 0, this.forceResultsStaging!, 0, 16);
        device.queue.submit([encoder.finish()]);
        
        await this.forceResultsStaging!.mapAsync(GPUMapMode.READ);
        const mapped = new Int32Array(this.forceResultsStaging!.getMappedRange());
        const results: number[] = [];
        for (let i = 0; i < numResults; i++) {
            results.push(mapped[i] / precisionScale);
        }
        this.forceResultsStaging!.unmap();
        return results;
    }

    public destroy(): void {
        const b1 = this.forceResultsBuffer;
        if (b1) b1.destroy();
        const b2 = this.forceResultsStaging;
        if (b2) b2.destroy();
    }
}

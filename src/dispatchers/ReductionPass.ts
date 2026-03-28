import { IVirtualGrid } from '../topology/GridAbstractions';
import { MasterBuffer } from '../memory/MasterBuffer';
import { ParityManager } from '../ParityManager';
import { PipelineCache } from './PipelineCache';
import { UniformStagingManager } from './UniformStagingManager';

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
        getWgslHeader: (type: string) => string,
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
        
        const pipeline = await pipelineCache.getPipeline(device, 'Reduction', source, getWgslHeader);
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
            const strideRow = buffer.layout?.strideRow || (dim.nx + 2);

            stagingBuffer[base + 0] = dim.nx;
            stagingBuffer[base + 1] = dim.ny;
            stagingBuffer[base + 2] = strideRow;
            stagingBuffer[base + 3] = dim.ny + (ghosts * 2);
            stagingBuffer[base + 5] = parityManager.currentTick;
            stagingBuffer[base + 6] = buffer.strideFace;
            stagingBuffer[base + 7] = faceMappings.length;

            const configParams = vGrid.config.params || {};
            for (let p = 0; p < 7; p++) {
                const key = `p${p}`;
                if (configParams[key] !== undefined) f32View[base + 8 + p] = configParams[key];
            }

            // FIX: Parameterized scaling for atomic precision
            f32View[base + 15] = precisionScale;

            for (let f = 0; f < 16; f++) {
                if (f < faceMappings.length) {
                    const m = faceMappings[f];
                    const res = parityManager.getFaceIndices(m.name);
                    stagingBuffer[base + 16 + f] = m.isPingPong ? res.read : res.base;
                } else {
                    stagingBuffer[base + 16 + f] = 999;
                }
            }
        }
        device.queue.writeBuffer(stagingManager.uniformBuffer!, 0, stagingBuffer.buffer);

        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        for (let i = 0; i < dispatchMetadata.length; i++) {
            const meta = dispatchMetadata[i];
            const ruleOffset = 0; // Reduction usually happens on rule 0 or dedicated rule
            
            const bindGroup = device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: buffer.gpuBuffer, offset: meta.dataOffset, size: meta.dataSize } },
                    { binding: 1, resource: { buffer: stagingManager.uniformBuffer!, offset: meta.uniformOffset + ruleOffset, size: bytesPerChunkAligned } },
                    { binding: 2, resource: { buffer: this.forceResultsBuffer } }
                ]
            });
            
            pass.setBindGroup(0, bindGroup);
            const dim = meta.vChunk.localDimensions;
            pass.dispatchWorkgroups(Math.ceil(dim.nx / 16), Math.ceil(dim.ny / 16), dim.nz || 1);
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
        if (this.forceResultsBuffer) this.forceResultsBuffer.destroy();
        if (this.forceResultsStaging) this.forceResultsStaging.destroy();
    }
}

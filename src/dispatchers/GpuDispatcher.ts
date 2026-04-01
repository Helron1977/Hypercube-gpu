import { IVirtualGrid } from '../topology/GridAbstractions';
import { MasterBuffer } from '../memory/MasterBuffer';
import { ParityManager } from '../ParityManager';
import { HypercubeGPUContext } from '../gpu/HypercubeGPUContext';
import { PipelineCache, PipelineMetadata } from './PipelineCache';
import { UniformStagingManager } from './UniformStagingManager';
import { HaloExchangePass } from './HaloExchangePass';
import { ReductionPass } from './ReductionPass';
import { WgslHeaderGenerator, WgslHeaderResult } from './WgslHeaderGenerator';

/**
 * Orchestrates simulation dispatch on the GPU using WebGPU.
 * v6.0: Refactored for Zero-Stall performance (No string operations in hot path).
 */
export class GpuDispatcher {
    public device: GPUDevice;
    private pipelineCache: PipelineCache = new PipelineCache();
    private stagingManager: UniformStagingManager;
    private haloExchange: HaloExchangePass = new HaloExchangePass();
    private reductionPass: ReductionPass = new ReductionPass();
    private bindGroupCache: Map<string, GPUBindGroup> = new Map();
    private activeMetadata: (PipelineMetadata | null)[] = [];
    private needsGlobalClear: boolean = false;

    constructor(
        private vGrid: IVirtualGrid,
        private buffer: MasterBuffer,
        private parityManager: ParityManager,
        mockDevice?: any
    ) {
        if (mockDevice) {
            this.device = mockDevice;
        } else {
            if (!HypercubeGPUContext.isInitialized) {
                throw new Error("GpuDispatcher: GPU Context not initialized.");
            }
            this.device = HypercubeGPUContext.device;
        }
        
        this.stagingManager = new UniformStagingManager(vGrid, buffer, parityManager);
        if (typeof window !== 'undefined') (window as any).dispatcher = this;
    }

    public get stagingBuffer(): Uint32Array | null {
        return this.stagingManager.stagingBuffer || null;
    }

    public invalidateActiveMetadata(): void {
        this.activeMetadata = [];
        this.bindGroupCache.clear();
    }

    public precomputeDispatchFlags(): void {
        const globals = this.vGrid.dataContract.getGlobalMappings();
        this.needsGlobalClear = globals.length > 0 && globals.some(g => g.type.startsWith('atomic'));
    }

    public async prepareKernels(kernels: Record<string, string>): Promise<void> {
        const descriptor = this.vGrid.dataContract.descriptor;
        const rules = descriptor.rules || [];
        
        // 1. Resolve core pipelines
        const promises = rules.map((scheme: any) => {
            const source = kernels[scheme.type];
            return source ? this.getPipeline(scheme.type, source, scheme.entryPoint || 'main') : Promise.resolve(null);
        });
        this.activeMetadata = await Promise.all(promises);

        // 2. Resolve HaloExchange if applicable
        if (this.vGrid.chunks.length > 1 && kernels['HaloExchange']) {
            await this.getPipeline('HaloExchange', kernels['HaloExchange']);
        }
    }

    public async getPipeline(type: string, source: string, entryPoint: string = 'main'): Promise<PipelineMetadata> {
        return this.pipelineCache.getPipeline(this.device, type, source, (t: string, s: string) => this.getWgslHeader(t, s), entryPoint);
    }

    public async dispatch(
        t: number = 0, 
        kernels: Record<string, string>, 
        overrides?: Record<string, number>,
        encoder?: GPUCommandEncoder
    ): Promise<GPUCommandEncoder> {
        const commandEncoder = encoder || this.device.createCommandEncoder();
        this.stagingManager.ensureBuffers(this.device);

        const descriptor = this.vGrid.dataContract.descriptor;
        const rules = descriptor.rules || [];
        const faceMappings = this.vGrid.dataContract.getFaceMappings();

        // 1. Prepare Pipelines (Zero-Stall: Use pre-computed metadata or fallback for transient kernels)
        let activeMetadata = this.activeMetadata;
        if (activeMetadata.length === 0 || kernels !== (this as any)._lastKernels) {
             const pipelinePromises = rules.map((scheme: any) => {
                const source = kernels[scheme.type];
                return source ? this.pipelineCache.getPipeline(this.device, scheme.type, source, (t: string, s: string) => this.getWgslHeader(t, s), scheme.entryPoint || 'main') : Promise.resolve(null);
            });
            activeMetadata = await Promise.all(pipelinePromises);
            (this as any)._lastKernels = kernels; 
        }

        // 2. Clear Global Workspace (Zero-Stall: Use pre-computed flag)
        if (this.needsGlobalClear && this.buffer.gpuGlobalBuffer) {
            commandEncoder.clearBuffer(this.buffer.gpuGlobalBuffer);
        }

        // 3. Update Uniform Staging
        this.stagingManager.updateStaging(t, rules, faceMappings, overrides);
        this.device.queue.writeBuffer(this.stagingManager.uniformBuffer!, 0, this.stagingManager.stagingBuffer!.buffer);

        // 3. Dispatch Compute Passes
        const stagingMetadata = this.stagingManager.getMetadata();
        const bytesPerChunkAligned = this.stagingManager.getBytesPerChunkAligned();

        for (let rIdx = 0; rIdx < rules.length; rIdx++) {
            const metaObj = activeMetadata[rIdx];
            if (!metaObj) continue;

            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(metaObj.pipeline);
            
            for (let i = 0; i < stagingMetadata.length; i++) {
                const meta = stagingMetadata[i];
                const ruleOffset = rIdx * bytesPerChunkAligned;

                // ZERO-STALL DISPATCH: Uses cached booleans from PipelineMetadata
                const bindGroup = this.getCachedBindGroup(
                    metaObj.pipeline, 
                    this.buffer.gpuBuffer, 
                    this.stagingManager.uniformBuffer!,
                    meta.globalIdx, 
                    meta.uniformOffset + ruleOffset, 
                    meta.dataOffset, 
                    meta.dataSize,
                    metaObj.usesAtomics, 
                    metaObj.usesGlobals
                );
                
                passEncoder.setBindGroup(0, bindGroup);
                const dim = meta.vChunk.localDimensions;
                if (dim.nz && dim.nz > 1) {
                    passEncoder.dispatchWorkgroups(Math.ceil(dim.nx / 8), Math.ceil(dim.ny / 8), Math.ceil(dim.nz / 4));
                } else {
                    passEncoder.dispatchWorkgroups(Math.ceil(dim.nx / 16), Math.ceil(dim.ny / 16), 1);
                }
            }
            passEncoder.end();
        }

        // 4. Synchronize Ghost Cells
        if (this.vGrid.chunks.length > 1) {
            await this.haloExchange.execute(
                this.device, this.vGrid, this.buffer, stagingMetadata, 
                this.pipelineCache, (t: string, s: string) => this.getWgslHeader(t, s), kernels, commandEncoder
            );
        }

        if (!encoder) {
            this.device.queue.submit([commandEncoder.finish()]);
        }
        return commandEncoder;
    }

    private getCachedBindGroup(
        pipeline: GPUComputePipeline,
        dataBuffer: GPUBuffer,
        uniformBuffer: GPUBuffer,
        globalChunkIdx: number,
        uniformOffset: number,
        dataOffset: number,
        chunkBufferSize: number,
        usesAtomics: boolean,
        usesGlobals: boolean
    ): GPUBindGroup {
        // Incorporate offsets into the cache key to ensure uniqueness if buffers are shared
        const key = `bg_${pipeline.label}_${globalChunkIdx}_${uniformOffset}_${dataOffset}_${this.buffer.byteOffset}_${this.buffer.atomicOffset}_${this.buffer.globalOffset}_${usesAtomics}_${usesGlobals}`;
        let bg = this.bindGroupCache.get(key);
        if (bg) return bg;

        const bytesPerChunkAligned = this.stagingManager.getBytesPerChunkAligned();
        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: dataBuffer, offset: this.buffer.byteOffset + dataOffset, size: chunkBufferSize } },
            { binding: 1, resource: { buffer: uniformBuffer, offset: uniformOffset, size: bytesPerChunkAligned } }
        ];

        if (this.buffer.gpuAtomicBuffer && usesAtomics) {
            entries.push({ 
                binding: 2, 
                resource: { 
                    buffer: this.buffer.gpuAtomicBuffer, 
                    offset: this.buffer.atomicOffset, 
                    size: Math.max(this.buffer.layout.atomicByteLength, 16) 
                } 
            });
        }

        if (usesGlobals && this.vGrid.dataContract.getGlobalMappings().length > 0) {
            if (this.buffer.gpuGlobalBuffer) {
                const globalSize = this.vGrid.dataContract.calculateGlobalBytes();
                entries.push({
                    binding: 3,
                    resource: { 
                        buffer: this.buffer.gpuGlobalBuffer, 
                        offset: this.buffer.globalOffset, 
                        size: Math.max(globalSize, 16) 
                    }
                });
            }
        }

        bg = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries
        });
        this.bindGroupCache.set(key, bg);
        return bg;
    }

    public getWgslHeader(ruleType: string, source: string = ""): WgslHeaderResult {
        const faceMappings = this.vGrid.dataContract.getFaceMappings();
        const globalMappings = this.vGrid.dataContract.getGlobalMappings();
        const paramNames = this.stagingManager.paramNames || [];
        return WgslHeaderGenerator.getHeader(ruleType, faceMappings, globalMappings, source, paramNames);
    }

    public async dispatchOneShot(
        combinedSource: string, 
        kernelName: string, 
        overrides?: Record<string, number>,
        entryPoint: string = 'main'
    ): Promise<void> {
        const metaObj = await this.getPipeline(kernelName, combinedSource, entryPoint);
        const commandEncoder = this.device.createCommandEncoder();

        // One-shot dispatch uses the current staging values but can apply overrides
        const faceMappings = this.vGrid.dataContract.getFaceMappings();
        const rules = this.vGrid.dataContract.descriptor.rules || [];
        this.stagingManager.updateStaging(0, rules, faceMappings, overrides);
        this.device.queue.writeBuffer(this.stagingManager.uniformBuffer!, 0, this.stagingManager.stagingBuffer!.buffer);

        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(metaObj.pipeline);

        const stagingMetadata = this.stagingManager.getMetadata();
        for (let i = 0; i < stagingMetadata.length; i++) {
            const meta = stagingMetadata[i];
            const bindGroup = this.getCachedBindGroup(
                metaObj.pipeline, this.buffer.gpuBuffer, this.stagingManager.uniformBuffer!,
                meta.globalIdx, meta.uniformOffset, meta.dataOffset, meta.dataSize,
                metaObj.usesAtomics, metaObj.usesGlobals
            );
            passEncoder.setBindGroup(0, bindGroup);
            const dim = meta.vChunk.localDimensions;
            if (dim.nz && dim.nz > 1) {
                passEncoder.dispatchWorkgroups(Math.ceil(dim.nx / 8), Math.ceil(dim.ny / 8), Math.ceil(dim.nz / 4));
            } else {
                passEncoder.dispatchWorkgroups(Math.ceil(dim.nx / 16), Math.ceil(dim.ny / 16), 1);
            }
        }
        passEncoder.end();
        this.device.queue.submit([commandEncoder.finish()]);
    }

    public async reduce(kernelName: string, source: string, faceName?: string, numResults: number = 1, precisionScale: number = 1e9): Promise<number[]> {
        return this.reductionPass.reduce(
            this.device, this.vGrid, this.buffer, this.parityManager,
            this.pipelineCache, this.stagingManager, (t: string, s: string) => this.getWgslHeader(t, s),
            source, faceName, numResults, precisionScale
        );
    }

    public destroy(): void {
        this.stagingManager.destroy();
        this.reductionPass.destroy();
        this.haloExchange.destroy();
        this.pipelineCache.clear();
        this.bindGroupCache.clear();
    }
}

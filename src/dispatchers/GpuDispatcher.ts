import { IVirtualGrid } from '../topology/GridAbstractions';
import { MasterBuffer } from '../memory/MasterBuffer';
import { ParityManager } from '../ParityManager';
import { HypercubeGPUContext } from '../gpu/HypercubeGPUContext';
import { PipelineCache } from './PipelineCache';
import { UniformStagingManager } from './UniformStagingManager';
import { HaloExchangePass } from './HaloExchangePass';
import { ReductionPass } from './ReductionPass';

/**
 * Orchestrates simulation dispatch on the GPU using WebGPU.
 * Refactored in v5.0.3 to use specialized modular passes.
 * Resolves technical debt of the 'God Object' by delegating core sub-tasks.
 */
export class GpuDispatcher {
    public device: GPUDevice;
    private pipelineCache: PipelineCache = new PipelineCache();
    private stagingManager: UniformStagingManager;
    private haloExchange: HaloExchangePass = new HaloExchangePass();
    private reductionPass: ReductionPass = new ReductionPass();
    private bindGroupCache: Map<string, GPUBindGroup> = new Map();

    /** @internal - Proxy for existing test suite */
    public get stagingBuffer(): Uint32Array | undefined { return this.stagingManager.stagingBuffer; }
    /** @internal - Proxy for existing test suite */
    public get uniformBuffer(): GPUBuffer | undefined { return this.stagingManager.uniformBuffer; }
    /** @internal - Proxy for existing test suite */
    public get bytesPerChunkAligned(): number { return this.stagingManager.getBytesPerChunkAligned(); }
    /** @internal - Proxy for existing test suite */
    public async getPipeline(type: string, source: string): Promise<GPUComputePipeline> { 
        return this.pipelineCache.getPipeline(this.device, type, source, (t) => this.getWgslHeader(t)); 
    }

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

    /**
     * Pre-warms kernels to avoid stalls during simulation.
     */
    public async prepareKernels(kernels: Record<string, string>): Promise<void> {
        const promises = Object.entries(kernels).map(([type, source]) =>
            this.pipelineCache.getPipeline(this.device, type, source, (t) => this.getWgslHeader(t))
        );
        await Promise.all(promises);
    }

    /**
     * Runs a simulation step (Compute Passes + Halo Exchange).
     * @param overrides - Optional temporary p0-p7 parameters for this step.
     */
    public async dispatch(
        t: number = 0, 
        kernels: Record<string, string>, 
        overrides?: Record<string, number>,
        encoder?: GPUCommandEncoder
    ): Promise<GPUCommandEncoder> {
        const descriptor = this.vGrid.dataContract.descriptor;
        const commandEncoder = encoder || this.device.createCommandEncoder();

        this.stagingManager.ensureBuffers(this.device);
        const rules = descriptor.rules || [];
        const faceMappings = this.vGrid.dataContract.getFaceMappings();

        // 1. Prepare Pipelines (Cached & Hashed)
        const pipelinePromises = rules.map(scheme => {
            const source = kernels[scheme.type];
            return source ? this.pipelineCache.getPipeline(this.device, scheme.type, source, (t) => this.getWgslHeader(t)) : Promise.resolve(null);
        });
        const activePipelines = await Promise.all(pipelinePromises);

        // 2. Update Uniform Staging & Transfer to GPU
        this.stagingManager.updateStaging(t, rules, faceMappings, overrides);
        this.device.queue.writeBuffer(this.stagingManager.uniformBuffer!, 0, this.stagingManager.stagingBuffer!.buffer);

        // 3. Dispatch Compute Passes for each Rule
        const metadata = this.stagingManager.getMetadata();
        const bytesPerChunkAligned = this.stagingManager.getBytesPerChunkAligned();

        for (let rIdx = 0; rIdx < rules.length; rIdx++) {
            const pipeline = activePipelines[rIdx];
            if (!pipeline) continue;

            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(pipeline);
            for (let i = 0; i < metadata.length; i++) {
                const meta = metadata[i];
                const ruleOffset = rIdx * bytesPerChunkAligned;
                
                const bindGroup = this.getCachedBindGroup(
                    pipeline, this.buffer.gpuBuffer, this.stagingManager.uniformBuffer!,
                    meta.globalIdx, meta.uniformOffset + ruleOffset, meta.dataOffset, meta.dataSize
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

        // 4. Synchronize Ghost Cells (Halos)
        if (this.vGrid.chunks.length > 1) {
            await this.haloExchange.execute(
                this.device, this.vGrid, this.buffer, metadata, 
                this.pipelineCache, (t) => this.getWgslHeader(t), kernels, commandEncoder
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
        chunkBufferSize: number
    ): GPUBindGroup {
        // Protection against Device Refresh
        if (this.device !== HypercubeGPUContext.device) {
            this.device = HypercubeGPUContext.device;
            this.bindGroupCache.clear();
            this.pipelineCache.clear();
        }

        const pipelineLabel = pipeline.label || 'Default';
        const bytesPerChunkAligned = this.stagingManager.getBytesPerChunkAligned();
        const key = `bg_${pipelineLabel}_${globalChunkIdx}_${uniformOffset}_${dataOffset}`;
        
        let bg = this.bindGroupCache.get(key);
        if (bg) return bg;

        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: dataBuffer, offset: dataOffset, size: chunkBufferSize } },
            { binding: 1, resource: { buffer: uniformBuffer, offset: uniformOffset, size: bytesPerChunkAligned } }
        ];

        if (this.buffer.gpuAtomicBuffer) {
            entries.push({ 
                binding: 2, 
                resource: { 
                    buffer: this.buffer.gpuAtomicBuffer, 
                    offset: this.buffer.layout.getFaceOffset(0, 0, 0), // Use start of atomic section
                    size: this.buffer.layout.atomicByteLength 
                } 
            });
        }

        bg = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries
        });
        this.bindGroupCache.set(key, bg);
        return bg;
    }

    /**
     * Performs a GPU reduction (sum/min/max) using the ReductionPass module.
     */
    public async reduce(kernelName: string, source: string, faceName?: string, numResults: number = 1, precisionScale: number = 1000000000.0): Promise<number[]> {
        return this.reductionPass.reduce(
            this.device, this.vGrid, this.buffer, this.parityManager,
            this.pipelineCache, this.stagingManager, (t) => this.getWgslHeader(t),
            source, faceName, numResults, precisionScale
        );
    }

    /**
     * Generates the standard WGSL header with auto-injected macros.
     */
    public getWgslHeader(ruleType: string): string {
        const descriptor = this.vGrid.dataContract.descriptor;
        const rule = descriptor.rules.find(r => r.type === ruleType);
        const explicitFaces = rule?.faces || [];
        const faceMappings = this.vGrid.dataContract.getFaceMappings();
        const hasAtomics = faceMappings.some(fm => (fm.type || 'scalar').startsWith('atomic'));
        const hasAtomicF32 = faceMappings.some(fm => fm.type === 'atomic_f32');

        let header = `
struct Uniforms {
  nx: u32, ny: u32, strideRow: u32, physicalNy: u32,
  t: f32, tick: u32, strideFace: u32, numFaces: u32,
  p0: f32, p1: f32, p2: f32, p3: f32,
  p4: f32, p5: f32, p6: f32, p7: f32,
  faces: array<u32, 64>,
  ghosts: u32,
  leftRole: u32, rightRole: u32, topRole: u32, bottomRole: u32, frontRole: u32, backRole: u32,
  reserved: u32
};
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;
${hasAtomics ? '@group(0) @binding(2) var<storage, read_write> dataAtomic: array<atomic<u32>>;' : ''}

fn getIndex(x: u32, y: u32) -> u32 { 
    return (y + uniforms.ghosts) * uniforms.strideRow + (x + uniforms.ghosts);
}

fn getIndex3D(x: u32, y: u32, z: u32) -> u32 { 
    return (z + uniforms.ghosts) * uniforms.strideRow * uniforms.physicalNy + (y + uniforms.ghosts) * uniforms.strideRow + (x + uniforms.ghosts);
}

${hasAtomicF32 ? `
fn _hypercube_atomicAddF32(ptr_idx: u32, val: f32) {
    var old = atomicLoad(&dataAtomic[ptr_idx]);
    loop {
        let expected = old;
        let newValue = bitcast<u32>(bitcast<f32>(expected) + val);
        let res = atomicCompareExchangeWeak(&dataAtomic[ptr_idx], expected, newValue);
        if (res.exchanged) { break; }
        old = res.old_value;
    }
}
` : ''}
\n`;

        const facesToGenerate = explicitFaces.length > 0 ? explicitFaces : faceMappings.map(fm => fm.name);

        for (let fIdx = 0; fIdx < Math.min(facesToGenerate.length, 64); fIdx++) {
            const rawName = facesToGenerate[fIdx];
            const faceName = rawName.split('.')[0];
            const suffix = rawName.includes('.') ? rawName.split('.')[1] : "";
            const m = faceMappings.find(fm => fm.name === faceName);
            if (!m) continue;

            const is3D = (this.vGrid.dimensions?.nz || 1) > 1;
            const isMultiComp = (m.numComponents || 1) > 1;
            const faceType = m.type || 'scalar';
            const isAtomic = faceType.startsWith('atomic');
            const isAtomicF32 = faceType === 'atomic_f32';
            
            const normalizedSuffix = suffix ? suffix.charAt(0).toUpperCase() + suffix.slice(1).toLowerCase() : "";
            const macroBaseName = normalizedSuffix ? faceName + "_" + normalizedSuffix : faceName;

            const generateBase = (id: string, f: number) => {
                const faceExpr = isMultiComp ? `(uniforms.faces[${f}] + u32(d))` : `uniforms.faces[${f}]`;
                const params2D = isMultiComp ? "x: u32, y: u32, d: u32" : "x: u32, y: u32";
                const params3D = isMultiComp ? "x: u32, y: u32, z: u32, d: u32" : "x: u32, y: u32, z: u32";
                const indexExpr2D = `${faceExpr} * uniforms.strideFace + (y + uniforms.ghosts) * uniforms.strideRow + (x + uniforms.ghosts)`;
                const indexExpr3D = `${faceExpr} * uniforms.strideFace + ((z + uniforms.ghosts) * uniforms.physicalNy + (y + uniforms.ghosts)) * uniforms.strideRow + (x + uniforms.ghosts)`;

                if (isAtomic) {
                    // Atomic Macros
                    if (isAtomicF32) {
                        header += `fn atomicAdd_${id}(${params2D}, val: f32) { _hypercube_atomicAddF32(${indexExpr2D}, val); }\n`;
                        header += `fn atomicAdd3D_${id}(${params3D}, val: f32) { _hypercube_atomicAddF32(${indexExpr3D}, val); }\n`;
                        header += `fn atomicLoad_${id}(${params2D}) -> f32 { return bitcast<f32>(atomicLoad(&dataAtomic[${indexExpr2D}])); }\n`;
                    } else {
                        header += `fn atomicAdd_${id}(${params2D}, val: u32) { atomicAdd(&dataAtomic[${indexExpr2D}], val); }\n`;
                        header += `fn atomicAdd3D_${id}(${params3D}, val: u32) { atomicAdd(&dataAtomic[${indexExpr3D}], val); }\n`;
                        header += `fn atomicLoad_${id}(${params2D}) -> u32 { return atomicLoad(&dataAtomic[${indexExpr2D}]); }\n`;
                    }
                } else {
                    // Standard Macros
                    header += `fn _read2D_${id}(${params2D}) -> f32 { return data[${indexExpr2D}]; }\n`;
                    header += `fn _write2D_${id}(${params2D}, val: f32) { data[${indexExpr2D}] = val; }\n`;
                    header += `fn _read3D_${id}(${params3D}) -> f32 { return data[${indexExpr3D}]; }\n`;
                    header += `fn _write3D_${id}(${params3D}, val: f32) { data[${indexExpr3D}] = val; }\n`;
                }
            };

            generateBase(macroBaseName, fIdx);

            const defineUserMacro = (alias: string, internalId: string, type: 'read' | 'write' | 'both') => {
                if (isAtomic) return; // Users call atomicAdd_ directly
                const args = isMultiComp ? (is3D ? "x, y, z, d" : "x, y, d") : (is3D ? "x, y, z" : "x, y");
                const params = isMultiComp ? (is3D ? "x: u32, y: u32, z: u32, d: u32" : "x: u32, y: u32, d: u32") : (is3D ? "x: u32, y: u32, z: u32" : "x: u32, y: u32");
                if (type === 'read' || type === 'both') header += `fn read_${alias}(${params}) -> f32 { return ${(is3D ? `_read3D_${internalId}` : `_read2D_${internalId}`)}(${args}); }\n`;
                if (type === 'write' || type === 'both') header += `fn write_${alias}(${params}, val: f32) { ${(is3D ? `_write3D_${internalId}` : `_write2D_${internalId}`)}(${args}, val); }\n`;
            };

            defineUserMacro(macroBaseName, macroBaseName, 'both');
            if (suffix === 'read' || suffix === 'now') defineUserMacro(faceName, macroBaseName, 'read');
            else if (suffix === 'write' || suffix === 'next') defineUserMacro(faceName, macroBaseName, 'write');
        }
        return header;
    }

    public destroy(): void {
        this.stagingManager.destroy();
        this.reductionPass.destroy();
        this.haloExchange.destroy();
        this.pipelineCache.clear();
        this.bindGroupCache.clear();
    }
}

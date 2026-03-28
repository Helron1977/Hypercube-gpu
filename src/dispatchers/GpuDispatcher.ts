import { IVirtualGrid, VirtualChunk } from '../topology/GridAbstractions';
import { MasterBuffer } from '../memory/MasterBuffer';
import { ParityManager } from '../ParityManager';
import { HypercubeGPUContext } from '../gpu/HypercubeGPUContext';
import { DataContract } from '../DataContract';
import { TopologyResolver, ResolvedTopology } from '../topology/TopologyResolver';

/**
 * Metadata for a chunk to avoid redundant calculations in the hot loop.
 */
interface ChunkDispatchMetadata {
    vChunk: VirtualChunk;
    uniformOffset: number;
    dataOffset: number;
    dataSize: number;
    globalIdx: number;
    topology: ResolvedTopology;
}

/**
 * Orchestrates simulation dispatch on the GPU using WebGPU.
 */
export class GpuDispatcher {
    public device: GPUDevice;
    private pipelines: Map<string, GPUComputePipeline> = new Map();
    private uniformBuffer?: GPUBuffer;
    private reductionBuffer?: GPUBuffer;
    private forceResultsBuffer?: GPUBuffer;
    private forceResultsStaging?: GPUBuffer;
    private haloTransferBuffer?: GPUBuffer;
    private topologyResolver: TopologyResolver = new TopologyResolver();
    private bindGroupCache: Map<string, GPUBindGroup> = new Map();
    private dispatchMetadata: ChunkDispatchMetadata[] = [];
    private bytesPerChunkAligned: number = 0;
    private stagingBuffer?: Uint32Array;

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
        this.bytesPerChunkAligned = HypercubeGPUContext.alignToUniform(512);
        this.refreshMetadata();
        if (typeof window !== 'undefined') (window as any).dispatcher = this;
    }

    private refreshMetadata(): void {
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

        for (let i = 0; i < this.dispatchMetadata.length; i++) {
            const meta = this.dispatchMetadata[i];
            const padding = this.vGrid.dataContract.descriptor.requirements.ghostCells || 0;
            const topo = meta.topology;

            for (let r = 0; r < maxRules; r++) {
                const base = (i * maxRules + r) * (this.bytesPerChunkAligned / 4);
                
                // Use GLOBAL layout values for strideRow and physicalNy to avoid tilts on multi-chunk grids
                const layoutStrideRow = this.buffer.layout?.strideRow || (meta.vChunk.localDimensions.nx + (padding * 2));
                const strideFace = this.buffer.layout?.strideFace || (this.buffer as any).strideFace || (layoutStrideRow * (meta.vChunk.localDimensions.ny + (padding * 2)));
                const layoutPhysicalNy = strideFace / layoutStrideRow; 

                this.stagingBuffer[base + 0] = meta.vChunk.localDimensions.nx;
                this.stagingBuffer[base + 1] = meta.vChunk.localDimensions.ny;
                this.stagingBuffer[base + 2] = layoutStrideRow;
                this.stagingBuffer[base + 3] = Math.floor(layoutPhysicalNy);
                
                // Index 80 is GHOSTS in WGSL struct
                this.stagingBuffer[base + 80] = padding;
                
                // Index 81-86 is ROLES in WGSL struct
                this.stagingBuffer[base + 81] = topo.leftRole;
                this.stagingBuffer[base + 82] = topo.rightRole;
                this.stagingBuffer[base + 83] = topo.topRole;
                this.stagingBuffer[base + 84] = topo.bottomRole;
                this.stagingBuffer[base + 85] = topo.frontRole;
                this.stagingBuffer[base + 86] = topo.backRole;
            }
        }
    }

    public async prepareKernels(kernels: Record<string, string>): Promise<void> {
        const promises = Object.entries(kernels).map(([type, source]) =>
            this.getPipeline(type, source)
        );
        await Promise.all(promises);
    }

    private ensureBuffers() {
        const maxRules = this.vGrid.config?.maxRules || 8;
        const totalUniformSize = this.vGrid.chunks.length * maxRules * this.bytesPerChunkAligned;
        if (!this.uniformBuffer || this.uniformBuffer.size < totalUniformSize) {
            if (this.uniformBuffer) this.uniformBuffer.destroy();
            this.uniformBuffer = this.device.createBuffer({
                size: totalUniformSize,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                label: 'GpuDispatcher Uniforms (Multi-Rule)'
            });
            this.bindGroupCache.clear();
        }
        if (!this.stagingBuffer) this.refreshMetadata();
    }

    public async dispatch(t: number = 0, kernels: Record<string, string>, encoder?: GPUCommandEncoder, extra?: { reductionBuffer?: GPUBuffer }): Promise<GPUCommandEncoder> {
        this.reductionBuffer = extra?.reductionBuffer;
        const descriptor = this.vGrid.dataContract.descriptor;
        const strideFace = this.buffer.strideFace;

        this.ensureBuffers();
        const u32Data = this.stagingBuffer!;
        const f32Data = new Float32Array(u32Data.buffer);
        const commandEncoder = encoder || this.device.createCommandEncoder();

        const rules = descriptor.rules || [];
        const faceMappings = this.vGrid.dataContract.getFaceMappings();

        // 1. Pré-chargement asynchrone des pipelines pour éviter les stalls en boucle
        const pipelinePromises = rules.map(scheme => {
            const source = kernels[scheme.type];
            return source ? this.getPipeline(scheme.type, source) : Promise.resolve(null);
        });
        const activePipelines = await Promise.all(pipelinePromises);

        // 2. Remplissage complet du staging buffer uniform (Multi-Rule)
        const maxRules = this.vGrid.config?.maxRules || 8;
        for (let rIdx = 0; rIdx < rules.length; rIdx++) {
            const scheme = rules[rIdx];
            if (!activePipelines[rIdx]) continue;

            for (let i = 0; i < this.dispatchMetadata.length; i++) {
                const base = (i * maxRules + rIdx) * (this.bytesPerChunkAligned / 4);
                const dim = this.dispatchMetadata[i].vChunk.localDimensions;
                const ghosts = descriptor.requirements.ghostCells || 0;

                // Use GLOBAL layout values for strideRow and physicalNy to avoid tilts on multi-chunk grids
                const layoutStrideRow = this.buffer.layout?.strideRow || (dim.nx + (ghosts * 2));
                const currentStrideFace = this.buffer.layout?.strideFace || (this.buffer as any).strideFace || (layoutStrideRow * (dim.ny + (ghosts * 2)));
                // physicalNy = ny + 2*ghosts (NOT strideFace/strideRow, which breaks for 3D)
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
                    let val = (scheme.params && scheme.params[key] !== undefined) ? scheme.params[key] : configParams[key];
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

                // Additional Professional Fields (Offset 80+)
                const boundaries = this.vGrid.config.boundaries || {};
                u32Data[base + 80] = ghosts;
                u32Data[base + 81] = (boundaries.left?.role === 'inflow') ? 3 : ((boundaries.left?.role === 'outflow') ? 4 : 2);
                u32Data[base + 82] = (boundaries.right?.role === 'inflow') ? 3 : ((boundaries.right?.role === 'outflow') ? 4 : 2);
                u32Data[base + 83] = (boundaries.top?.role === 'moving_wall') ? 10 : 2;
                u32Data[base + 84] = 2; // bottom
                u32Data[base + 85] = 2; // front
                u32Data[base + 86] = 2; // back
            }
        }

        // 3. TRANSFERT DES UNIFORMS AVANT TOUTES LES PASSES (Determinism Check)
        this.device.queue.writeBuffer(this.uniformBuffer!, 0, u32Data.buffer);

        // 4. Dispatch des Compute Passes
        for (let rIdx = 0; rIdx < rules.length; rIdx++) {
            const pipeline = activePipelines[rIdx];
            if (!pipeline) continue;

            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(pipeline);
            for (let i = 0; i < this.dispatchMetadata.length; i++) {
                const meta = this.dispatchMetadata[i];
                const ruleOffset = rIdx * this.bytesPerChunkAligned;
                const bindGroup = this.getCachedBindGroup(
                    pipeline, this.buffer.gpuBuffer, this.uniformBuffer!,
                    meta.globalIdx, meta.uniformOffset + ruleOffset, meta.dataOffset, meta.dataSize,
                    this.reductionBuffer
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

        if (this.vGrid.chunks.length > 1) {
            await this.syncHalos(kernels, commandEncoder);
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
        reductionBuffer?: GPUBuffer
    ): GPUBindGroup {
        // Sécurité : Si le device a changé (ex: context loss), vider le cache
        if (this.device !== HypercubeGPUContext.device) {
            this.device = HypercubeGPUContext.device;
            this.bindGroupCache.clear();
            this.pipelines.clear();
        }

        const pipelineLabel = pipeline.label || 'Default';
        const key = `bg_${pipelineLabel}_${globalChunkIdx}_${uniformOffset}_${dataOffset}_${reductionBuffer ? 'R' : 'N'}`;
        let bg = this.bindGroupCache.get(key);
        if (bg) return bg;

        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: dataBuffer, offset: dataOffset, size: chunkBufferSize } },
            { binding: 1, resource: { buffer: uniformBuffer, offset: uniformOffset, size: this.bytesPerChunkAligned } }
        ];
        if (reductionBuffer) {
            entries.push({ binding: 2, resource: { buffer: reductionBuffer } });
        }

        bg = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries
        });
        this.bindGroupCache.set(key, bg);
        return bg;
    }

    public getWgslHeader(ruleType: string): string {
        const descriptor = this.vGrid.dataContract.descriptor;
        const rule = descriptor.rules.find(r => r.type === ruleType);
        const explicitFaces = rule?.faces || [];
        const faceMappings = this.vGrid.dataContract.getFaceMappings();

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

fn getIndex(x: u32, y: u32) -> u32 { 
    return (y + uniforms.ghosts) * uniforms.strideRow + (x + uniforms.ghosts);
}

fn getIndex3D(x: u32, y: u32, z: u32) -> u32 { 
    return (z + uniforms.ghosts) * uniforms.strideRow * uniforms.physicalNy + (y + uniforms.ghosts) * uniforms.strideRow + (x + uniforms.ghosts);
}
\n`;

        // Face-specific helper functions (Macros)
        const facesToGenerate = explicitFaces.length > 0 ? explicitFaces : faceMappings.map(fm => fm.name);

        for (let fIdx = 0; fIdx < Math.min(facesToGenerate.length, 64); fIdx++) {
            const rawName = facesToGenerate[fIdx];
            const faceName = rawName.split('.')[0];
            const suffix = rawName.includes('.') ? rawName.split('.')[1] : "";
            const m = faceMappings.find(fm => fm.name === faceName);
            if (!m) continue;

            const is3D = (this.vGrid.dimensions?.nz || 1) > 1;
            const isMultiComp = (m.numComponents || 1) > 1;
            const normalizedSuffix = suffix ? suffix.charAt(0).toUpperCase() + suffix.slice(1).toLowerCase() : "";
            const macroBaseName = normalizedSuffix ? faceName + "_" + normalizedSuffix : faceName;

            // 1. Internal Base Macros (Slot-Specific)
            const generateBase = (id: string, f: number, slotOffset: string) => {
                // For multi-component faces (LBM D2Q9), component d offsets by full strideFace planes
                const faceExpr = isMultiComp ? `(uniforms.faces[${f}] + u32(d))` : `uniforms.faces[${f}]`;
                const params2D = isMultiComp ? "x: u32, y: u32, d: u32" : "x: u32, y: u32";
                const params3D = isMultiComp ? "x: u32, y: u32, z: u32, d: u32" : "x: u32, y: u32, z: u32";

                header += `fn _read2D_${id}(${params2D}) -> f32 { return data[${faceExpr} * uniforms.strideFace + (y + uniforms.ghosts) * uniforms.strideRow + (x + uniforms.ghosts)]; }\n`;
                header += `fn _write2D_${id}(${params2D}, val: f32) { data[${faceExpr} * uniforms.strideFace + (y + uniforms.ghosts) * uniforms.strideRow + (x + uniforms.ghosts)] = val; }\n`;
                header += `fn _read3D_${id}(${params3D}) -> f32 { return data[${faceExpr} * uniforms.strideFace + ((z + uniforms.ghosts) * uniforms.physicalNy + (y + uniforms.ghosts)) * uniforms.strideRow + (x + uniforms.ghosts)]; }\n`;
                header += `fn _write3D_${id}(${params3D}, val: f32) { data[${faceExpr} * uniforms.strideFace + ((z + uniforms.ghosts) * uniforms.physicalNy + (y + uniforms.ghosts)) * uniforms.strideRow + (x + uniforms.ghosts)] = val; }\n`;
            };

            generateBase(macroBaseName, fIdx, "");

            // 2. Public Facing Macros
            const defineUserMacro = (alias: string, internalId: string, type: 'read' | 'write' | 'both') => {
                const args = isMultiComp ? (is3D ? "x, y, z, d" : "x, y, d") : (is3D ? "x, y, z" : "x, y");
                const params = isMultiComp ? (is3D ? "x: u32, y: u32, z: u32, d: u32" : "x: u32, y: u32, d: u32") : (is3D ? "x: u32, y: u32, z: u32" : "x: u32, y: u32");
                
                if (type === 'read' || type === 'both') {
                    header += `fn read_${alias}(${params}) -> f32 { return ${(is3D ? `_read3D_${internalId}` : `_read2D_${internalId}`)}(${args}); }\n`;
                }
                if (type === 'write' || type === 'both') {
                    header += `fn write_${alias}(${params}, val: f32) { ${(is3D ? `_write3D_${internalId}` : `_write2D_${internalId}`)}(${args}, val); }\n`;
                }
            };

            // Main Macro for this slot
            defineUserMacro(macroBaseName, macroBaseName, 'both');

            // Compatibility Aliases: only needed when suffix makes macroBaseName differ from faceName
            if (suffix === 'read' || suffix === 'now') {
                defineUserMacro(faceName, macroBaseName, 'read');
            } else if (suffix === 'write' || suffix === 'next') {
                defineUserMacro(faceName, macroBaseName, 'write');
            }
            // No else: when suffix is empty, macroBaseName === faceName, already defined above.
        }

        return header;
    }

    private async getPipeline(ruleType: string, source: string): Promise<GPUComputePipeline> {
        // Sécurité Contextuelle : Si le device a changé, on doit TOUT purger
        if (this.device !== HypercubeGPUContext.device) {
            this.device = HypercubeGPUContext.device;
            this.bindGroupCache.clear();
            this.pipelines.clear();
        }

        const header = this.getWgslHeader(ruleType);
        const combinedSource = header + source;
        const cacheKey = `${ruleType}_${combinedSource.length}_${combinedSource.substring(0, 32)}`;

        let p = this.pipelines.get(cacheKey);
        if (p) return p;

        p = await HypercubeGPUContext.createComputePipelineAsync(combinedSource, `Kernel_${ruleType}`);
        this.pipelines.set(cacheKey, p);
        return p;
    }

    public async reduce(kernelName: string, source: string, faceName?: string, numResults: number = 1): Promise<number[]> {
        if (!this.forceResultsBuffer) {
            this.forceResultsBuffer = this.device.createBuffer({
                size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
            });
            this.forceResultsStaging = this.device.createBuffer({
                size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
            });
        }
        this.device.queue.writeBuffer(this.forceResultsBuffer, 0, new Int32Array(4).fill(0));
        const pipeline = await this.getPipeline('Reduction', source);
        const encoder = this.device.createCommandEncoder();
        this.ensureBuffers();
        const faceMappings = this.vGrid.dataContract.getFaceMappings();
        const f32View = new Float32Array(this.stagingBuffer!.buffer);
        const maxRules = this.vGrid.config?.maxRules || 8;

        for (let i = 0; i < this.dispatchMetadata.length; i++) {
            const base = (i * maxRules) * (this.bytesPerChunkAligned / 4);
            const dim = this.dispatchMetadata[i].vChunk.localDimensions;
            const strideRow = this.buffer.layout?.strideRow || (dim.nx + 2);
            this.stagingBuffer![base + 0] = dim.nx;
            this.stagingBuffer![base + 1] = dim.ny;
            const ghosts = this.vGrid.dataContract.descriptor.requirements.ghostCells || 1;
            this.stagingBuffer![base + 2] = strideRow;
            this.stagingBuffer![base + 3] = dim.ny + (ghosts * 2);
            this.stagingBuffer![base + 5] = this.parityManager.currentTick;
            this.stagingBuffer![base + 6] = this.buffer.strideFace;
            this.stagingBuffer![base + 7] = faceMappings.length;

            // On passe dynamiquement l'index de la face 'obstacle' et 'f'
            const obsIdx = faceMappings.findIndex(f => f.name === 'obstacle');
            const configParams = this.vGrid.config.params || {};
            for (let p = 0; p < 7; p++) {
                const key = `p${p}`;
                if (configParams[key] !== undefined) f32View[base + 8 + p] = configParams[key];
            }

            // Calcul du facteur d'échelle p7
            // Facteur global de réduction (Scaling) pour la précision atomique (Int32)
            f32View[base + 15] = (faceName) ? 10000.0 : 1000000000.0;

            // Mapping STRICT du DataContract vers les slots f0...f15
            for (let f = 0; f < 16; f++) {
                if (f < faceMappings.length) {
                    const m = faceMappings[f];
                    const res = this.parityManager.getFaceIndices(m.name);
                    // Dans une réduction ponctuelle, on préfère lire la face 'read' par défaut si ping-pong
                    this.stagingBuffer![base + 16 + f] = m.isPingPong ? res.read : res.base;
                } else {
                    this.stagingBuffer![base + 16 + f] = 999;
                }
            }
        }
        this.device.queue.writeBuffer(this.uniformBuffer!, 0, this.stagingBuffer!.buffer);

        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        for (let i = 0; i < this.dispatchMetadata.length; i++) {
            const meta = this.dispatchMetadata[i];
            const bindGroup = this.getCachedBindGroup(pipeline, this.buffer.gpuBuffer, this.uniformBuffer!, meta.globalIdx, meta.uniformOffset, meta.dataOffset, meta.dataSize, this.forceResultsBuffer);
            pass.setBindGroup(0, bindGroup);
            const dim = meta.vChunk.localDimensions;
            pass.dispatchWorkgroups(Math.ceil(dim.nx / 16), Math.ceil(dim.ny / 16), dim.nz || 1);
        }
        pass.end();
        encoder.copyBufferToBuffer(this.forceResultsBuffer, 0, this.forceResultsStaging!, 0, 16);
        this.device.queue.submit([encoder.finish()]);
        await this.forceResultsStaging!.mapAsync(GPUMapMode.READ);
        const mapped = new Int32Array(this.forceResultsStaging!.getMappedRange());
        const results: number[] = [];
        for (let i = 0; i < numResults; i++) results.push(mapped[i] / 1000000000.0);
        this.forceResultsStaging!.unmap();
        return results;
    }

    private async syncHalos(kernels: Record<string, string>, encoder: GPUCommandEncoder): Promise<void> {
        const source = kernels['HaloExchange'];
        if (!source) return;
        const pipeline = await this.getPipeline('HaloExchange', source);
        const transfers: any[] = [];
        const stride = this.buffer.strideFace;
        const numFaces = this.vGrid.dataContract.getFaceMappings().length;

        for (const meta of this.dispatchMetadata) {
            for (const joint of meta.vChunk.joints) {
                if (joint.role !== 'joint') continue;
                const neighbor = this.vGrid.chunks.find(c => c.id === joint.neighborId);
                if (!neighbor) continue;
                const neighborMeta = this.dispatchMetadata.find(m => m.vChunk.id === neighbor.id);
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
            this.haloTransferBuffer = this.device.createBuffer({
                size: transfers.length * transferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
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
        this.device.queue.writeBuffer(this.haloTransferBuffer, 0, rawTransfers);

        const passEncoder = encoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        const bindGroup = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffer.gpuBuffer } },
                { binding: 2, resource: { buffer: this.haloTransferBuffer } }
            ]
        });
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(16, 16, transfers.length);
        passEncoder.end();
    }

    public destroy(): void {
        if (this.uniformBuffer) this.uniformBuffer.destroy();
        if (this.reductionBuffer) this.reductionBuffer.destroy();
        if (this.forceResultsBuffer) this.forceResultsBuffer.destroy();
        if (this.forceResultsStaging) this.forceResultsStaging.destroy();
        if (this.haloTransferBuffer) this.haloTransferBuffer.destroy();
        this.pipelines.clear();
        this.bindGroupCache.clear();
    }
}

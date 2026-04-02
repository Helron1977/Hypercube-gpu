import { IVirtualGrid } from './topology/GridAbstractions';
import { MasterBuffer } from './memory/MasterBuffer';
import { GpuDispatcher } from './dispatchers/GpuDispatcher';
import { ParityManager } from './ParityManager';
import { HypercubeGPUContext } from './gpu/HypercubeGPUContext';
import { WgslHeaderResult } from './dispatchers/WgslHeaderGenerator';

/**
 * High-level wrapper for a Hypercube GPU simulation.
 * @typeparam TParams - The type of the simulation parameters (must be a record of numbers).
 */
export class GpuEngine<TParams extends Record<string, number> = any, TFaces = any> {
    private bufferPool: Map<string, Float32Array> = new Map();
    private registeredKernels: Record<string, string> = {};
    private persistentOverrides: Partial<TParams> = {};
    private compilationPending: Promise<void> | null = null;
    public readonly params: TParams & { set: (name: keyof TParams, value: number) => GpuEngine<TParams, TFaces>['params'] };

    constructor(
        public readonly vGrid: IVirtualGrid<TParams>,
        public readonly buffer: MasterBuffer,
        public readonly dispatcher: GpuDispatcher,
        public readonly parityManager: ParityManager
    ) {
        if (!HypercubeGPUContext.isInitialized) {
            throw new Error("[GpuEngine] GPU Context was not correctly initialized. Use GpuCoreFactory to create engine instances.");
        }

        // Fluent & Proxy Params Controller
        this.params = new Proxy(this.persistentOverrides, {
            get: (target, prop) => {
                if (prop === 'set') {
                    return (name: keyof TParams, value: number) => {
                        this.setParam(name, value);
                        return this.params;
                    };
                }
                return (target as any)[prop];
            },
            set: (target, prop, value) => {
                this.setParam(prop as any, value);
                return true;
            }
        }) as any;
    }

    /**
     * Injects data directly into the current READ slot of a face.
     * Unlike setFaceData, this is intended for mid-simulation interaction
     * and correctly identifies the active parity slot before staging.
     */
    public injectData(faceName: string, data: Float32Array | number[], chunkId: string = 'chunk_0_0_0'): void {
        const parity = this.parityManager.getFaceIndices(faceName).read;
        this.buffer.setFaceData(chunkId, faceName, data, false, parity);
        this.buffer.syncToDevice();
    }

    /**
     * Registers simulation kernels persistently.
     * Once registered, step() can be called without passing source code.
     */
    public async use(kernels: Record<string, string>): Promise<void> {
        this.registeredKernels = { ...this.registeredKernels, ...kernels };
        
        // v6.0: Trigger Zero-Stall Pre-compilation
        this.dispatcher.invalidateActiveMetadata();
        this.compilationPending = this.dispatcher.prepareKernels(this.registeredKernels);
        await this.compilationPending;
        this.dispatcher.precomputeDispatchFlags();
    }

    /**
     * Sets a persistent parameter override (e.g., 'viscosity', 'conductivity').
     * These overrides are merged into every subsequent step.
     */
    public setParam(name: keyof TParams, value: number): void {
        this.persistentOverrides[name] = value as any;
    }

    /**
     * Executes one or more simulation steps.
     * @param paramsOrCount Either the number of steps to run (number) or transient parameter overrides (TParams).
     * @param kernels Optional transient kernels (overrides registered ones for this call).
     */
    public async step(paramsOrCount: TParams | number = 1, kernels?: Record<string, string>): Promise<void> {
        if (this.compilationPending) {
            await this.compilationPending;
        }

        const activeKernels = kernels || this.registeredKernels;
        if (Object.keys(activeKernels).length === 0) {
            throw new Error("GpuEngine: No kernels registered. Use engine.use() before calling step().");
        }

        if (typeof paramsOrCount === 'number') {
            for (let i = 0; i < paramsOrCount; i++) {
                // Apply persistent overrides during multi-step runs
                await this.dispatcher.dispatch(this.parityManager.currentTick, activeKernels, this.persistentOverrides as TParams);
                this.parityManager.increment();
            }
        } else {
            // Merge transient overrides with persistent ones
            const mergedParams = { ...this.persistentOverrides, ...paramsOrCount } as TParams;
            await this.dispatcher.dispatch(this.parityManager.currentTick, activeKernels, mergedParams);
            this.parityManager.increment();
        }
    }

    /**
     * Synchronizes only parameters (p0-p7) and time to the GPU.
     * Use this in the loop for dynamic interactions without overwriting the field results.
     */
    public async syncParams(kernels: Record<string, string>): Promise<void> {
        await this.dispatcher.dispatch(this.parityManager.currentTick, kernels);
    }

    /**
     * Executes a one-shot kernel dispatch without advancing the simulation tick.
     * v6.0.5: Added for initializations, diagnostics, or impulse injections.
     */
    public async executeTransient(name: string, source: string, overrides?: Record<string, number>): Promise<void> {
        const header = this.dispatcher.getWgslHeader(name, source);
        const combined = header.code + source;
        await this.dispatcher.dispatchOneShot(combined, name, overrides);
    }

    public getWgslHeader(ruleType: string): WgslHeaderResult {
        return this.dispatcher.getWgslHeader(ruleType);
    }

    public setFaceData(chunkId: string, faceName: string, data: Float32Array | number[], fillAllPingPong: boolean = false, parity?: number): void {
        this.buffer.setFaceData(chunkId, faceName, data, fillAllPingPong, parity);
    }

    public getFaceData(chunkId: string, faceName: string, parity?: number): Float32Array {
        return this.buffer.getFaceData(chunkId, faceName, parity);
    }

    public async syncFacesToHost(faces: (string | number)[]): Promise<void> {
        await this.buffer.syncFacesToHost(faces);
    }

    /**
     * Contextual v6.0 SOTA: Synchronizes only the Simulation Globals (Binding 3) from GPU to Host.
     * @deprecated Use getGlobal() which leverages Direct-Read Manifest Architecture (v9).
     */
    public async syncGlobalsToHost(): Promise<void> {
        // No-op for v9 Direct-Read Architecture
    }

    /**
     * High-level API to retrieve a simulation-wide contextual variable.
     * V9: Direct-Read Architecture handles async GPU mapping without host pollution.
     */
    public async getGlobal(name: string): Promise<Float32Array | Int32Array> {
        return this.buffer.readGlobalData(name);
    }

    /**
     * Elite Helper: Synchronizes a single face and returns its inner data (no ghosts).
     * @param faceName The name of the face to retrieve.
     */
    public async getFace(faceName: string, parity?: number): Promise<Float32Array> {
        const resolvedParity = parity !== undefined ? parity : this.parityManager.getFaceIndices(faceName).read;
        await this.syncFacesToHost([faceName]);
        const fullData = this.getFaceData('chunk_0_0_0', faceName, resolvedParity);
        
        // If it's a scalar/reduction face (small length), return raw
        const { nx, ny } = this.vGrid.dimensions;
        if (fullData.length < nx * ny) {
            return fullData;
        }
        
        return this.getInnerFaceData('chunk_0_0_0', faceName);
    }

    public async syncToHost(): Promise<void> {
        await this.buffer.syncToHost();
    }

    /**
     * Returns a view of the data for a specific face, EXCLUDING ghost cells.
     * Useful for direct injection into rendering engines (Three.js, etc.)
     */
    public getInnerFaceData(chunkId: string, faceName: string, parity?: number): Float32Array {
        const resolvedParity = parity !== undefined ? parity : this.parityManager.getFaceIndices(faceName).read;
        const fullData = this.getFaceData(chunkId, faceName, resolvedParity);
        const ghosts = this.vGrid.config.engine === 'test-engine' ? 1 : (this.vGrid.dataContract.descriptor.requirements.ghostCells || 0); 
        
        if (ghosts === 0) return fullData;

        const { nx, ny, nz } = this.vGrid.dimensions;
        const lx = nx + (ghosts * 2);
        const ly = ny + (ghosts * 2);
        const lz = nz > 1 ? nz + (ghosts * 2) : 1;
        
        const poolKey = `${chunkId}:${faceName}:${nx}:${ny}:${lz}`;
        let innerData = this.bufferPool.get(poolKey);
        
        if (!innerData || innerData.length !== nx * ny * lz) {
            innerData = new Float32Array(nx * ny * lz);
            this.bufferPool.set(poolKey, innerData);
        }
        
        let destIdx = 0;
        
        for (let z = 0; z < lz; z++) {
            const zOff = (nz > 1) ? (z + ghosts) : 0;
            for (let y = 0; y < ny; y++) {
                const yOff = y + ghosts;
                const srcStart = (zOff * lx * ly) + (yOff * lx) + ghosts;
                innerData.set(fullData.subarray(srcStart, srcStart + nx), destIdx);
                destIdx += nx;
            }
        }
        
        return innerData;
    }

    public destroy(): void {
        this.buffer.destroy();
        this.dispatcher.destroy();
        this.bufferPool.clear();
    }

    /**
     * Resolves when the engine and its buffers are fully ready to be used.
     */
    public async ready(): Promise<void> {
        await HypercubeGPUContext.init();
        // Ensure initial sync of uniforms
        await this.syncParams({});
    }

    /**
     * Executes a global reduction (Sum) on a specific face.
     * v6.1: Dynamic scaling to prevent atomic i32 overflow.
     */
    public async reduceField(faceName: string, source: string): Promise<number> {
        const { nx, ny, nz } = this.vGrid.dimensions;
        const totalCells = nx * ny * (nz || 1);
        
        // Dynamic Safe Scaling (v6.1):
        // Scale to keep total sum < 2e9 (i32 limit). 
        // Max expected sum assumes rho=2.0 per cell for worst-case safety.
        const safeScale = Math.floor(2000000000 / (totalCells * 2.0));
        const precisionScale = Math.max(1, Math.min(1000000000, safeScale));
        
        const results = await this.dispatcher.reduce('Reduction', source, faceName, 1, precisionScale);
        return results[0];
    }

    /**
     * Measures aerodynamic forces using the Momentum Exchange Method (MEM).
     * Now uses the dedicated Global Workspace for zero-waste retrieval.
     */
    public async reduceForces(): Promise<[number, number]> {
        const results = await this.getGlobal('forces');
        const scale = (results instanceof Int32Array) ? (this.persistentOverrides.p7 || 1e9) : 1.0;
        return [results[0] / scale, results[1] / scale];
    }

    public syncToDevice(): void {
        this.buffer.syncToDevice();
    }
}

import { IVirtualGrid } from './topology/GridAbstractions';
import { MasterBuffer } from './memory/MasterBuffer';
import { GpuDispatcher } from './dispatchers/GpuDispatcher';
import { ParityManager } from './ParityManager';
import { HypercubeGPUContext } from './gpu/HypercubeGPUContext';

/**
 * High-level wrapper for a Hypercube GPU simulation.
 * @typeparam TParams - The type of the simulation parameters (must be a record of numbers).
 */
export class GpuEngine<TParams extends Record<string, number> = any, TFaces = any> {
    private bufferPool: Map<string, Float32Array> = new Map();
    private registeredKernels: Record<string, string> = {};
    private persistentOverrides: Partial<TParams> = {};

    constructor(
        public readonly vGrid: IVirtualGrid<TParams>,
        public readonly buffer: MasterBuffer,
        public readonly dispatcher: GpuDispatcher,
        public readonly parityManager: ParityManager
    ) {}

    /**
     * Registers simulation kernels persistently.
     * Once registered, step() can be called without passing source code.
     */
    public use(kernels: Record<string, string>): void {
        this.registeredKernels = { ...this.registeredKernels, ...kernels };
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

    public getWgslHeader(ruleType: string): string {
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
     * Elite Helper: Synchronizes a single face and returns its inner data (no ghosts).
     * @param faceName The name of the face to retrieve.
     */
    public async getFace(faceName: string): Promise<Float32Array> {
        await this.syncFacesToHost([faceName]);
        const fullData = this.getFaceData('chunk_0_0_0', faceName);
        
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
        const fullData = this.getFaceData(chunkId, faceName, parity);
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
     */
    public async reduceField(faceName: string, source: string): Promise<number> {
        const results = await this.dispatcher.reduce('Reduction', source, faceName, 1);
        return results[0];
    }

    /**
     * Measures aerodynamic forces using the Momentum Exchange Method (MEM).
     * @returns [Fx, Fy] in lattice units.
     */
    public async reduceForces(source: string): Promise<[number, number]> {
        const results = await this.dispatcher.reduce('Forces', source, undefined, 2);
        return [results[0], results[1]];
    }

    public syncToDevice(): void {
        this.buffer.syncToDevice();
    }
}

import { IVirtualGrid } from './topology/GridAbstractions';
import { MasterBuffer } from './memory/MasterBuffer';
import { GpuDispatcher } from './dispatchers/GpuDispatcher';
import { ParityManager } from './ParityManager';
import { HypercubeGPUContext } from './gpu/HypercubeGPUContext';

/**
 * High-level wrapper for a Hypercube GPU simulation.
 */
export class GpuEngine<TParams = any, TFaces = any> {
    constructor(
        public readonly vGrid: IVirtualGrid<TParams>,
        public readonly buffer: MasterBuffer,
        public readonly dispatcher: GpuDispatcher,
        public readonly parityManager: ParityManager
    ) {}

    public async step(kernels: Record<string, string>, count: number = 1): Promise<void> {
        for (let i = 0; i < count; i++) {
            await this.dispatcher.dispatch(this.parityManager.currentTick, kernels);
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
        // Note: For now we use the descriptor, but we can refine the filtering logic
        
        if (ghosts === 0) return fullData;

        const { nx, ny, nz } = this.vGrid.dimensions;
        const lx = nx + (ghosts * 2);
        const ly = ny + (ghosts * 2);
        const lz = nz > 1 ? nz + (ghosts * 2) : 1;
        
        const innerData = new Float32Array(nx * ny * lz);
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

    /**
     * Resolves when the engine and its buffers are fully ready to be used.
     */
    public async ready(): Promise<this> {
        // Ensure initial sync of uniforms
        await this.syncParams({});
        return this;
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

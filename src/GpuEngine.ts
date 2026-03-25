import { IVirtualGrid } from './topology/GridAbstractions';
import { MasterBuffer } from './memory/MasterBuffer';
import { GpuDispatcher } from './dispatchers/GpuDispatcher';
import { ParityManager } from './ParityManager';
import { HypercubeGPUContext } from './gpu/HypercubeGPUContext';

/**
 * High-level wrapper for a Hypercube GPU simulation.
 */
export class GpuEngine {
    constructor(
        public readonly vGrid: IVirtualGrid,
        public readonly buffer: MasterBuffer,
        public readonly dispatcher: GpuDispatcher,
        public readonly parityManager: ParityManager
    ) {}

    /**
     * Executes one or more simulation steps.
     * @param kernels Record of WGSL sources for the engine's rules.
     * @param count Number of steps to run.
     */
    public async step(kernels: Record<string, string>, count: number = 1): Promise<void> {
        for (let i = 0; i < count; i++) {
            await this.dispatcher.dispatch(i, kernels);
            this.parityManager.increment();
        }
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

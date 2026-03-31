import { HypercubeManifest } from './types';
import { GpuEngine } from './GpuEngine';
import { GpuCoreFactory } from './GpuCoreFactory';
import { ManifestValidator } from './builders/ManifestValidator';
import { SharedMasterBuffer } from './memory/SharedMasterBuffer';
import { VirtualGrid } from './topology/VirtualGrid';
import { MemoryLayout } from './memory/MemoryLayout';

/**
 * The "Blessed Path" for creating a Hypercube GPU Simulation.
 * Validates the manifest and builds the engine in one call.
 * 
 * @param manifest The complete simulation manifest (engine + config).
 * @param mockDevice Optional GPUDevice for testing/headless environments.
 * @returns A fully initialized GpuEngine.
 */
export async function createSimulation<TParams extends Record<string, number> = any, TFaces = any>(
    manifest: HypercubeManifest<TParams, TFaces>,
    mockDevice?: any
): Promise<GpuEngine<TParams, TFaces>> {
    // 1. Validate
    ManifestValidator.validate(manifest);

    // 2. Build
    const factory = new GpuCoreFactory();
    const engine = await factory.build(manifest.config, manifest.engine, mockDevice);

    return engine;
}

/**
 * Create a coupled simulation that shares physical memory with other engines.
 * 
 * @param manifest The simulation manifest.
 * @param shared High-level shared buffer manager.
 * @returns An engine linked to the shared GPU resources.
 */
export async function linkSimulation<TParams extends Record<string, number> = any, TFaces = any>(
    manifest: HypercubeManifest<TParams, TFaces>,
    shared: SharedMasterBuffer
): Promise<GpuEngine<TParams, TFaces>> {
    ManifestValidator.validate(manifest);

    const factory = new GpuCoreFactory();
    
    // Virtual calculation to determine required bytes
    const vGrid = new VirtualGrid(manifest.config, manifest.engine);
    const layout = new MemoryLayout(vGrid);
    
    const standardBytes = layout.standardByteLength;
    const atomicBytes = layout.atomicByteLength;
    const globalBytes = vGrid.dataContract.calculateGlobalBytes();
    
    // Allocate offsets from shared pool
    const byteOffset = shared.allocate(standardBytes, 'standard');
    const atomicOffset = atomicBytes > 0 ? shared.allocate(atomicBytes, 'atomic') : 0;
    const globalOffset = globalBytes > 0 ? shared.allocate(globalBytes, 'global') : 0;

    return await factory.build(manifest.config, manifest.engine, undefined, {
        gpuBuffer: shared.gpuBuffer,
        gpuAtomicBuffer: shared.gpuAtomicBuffer,
        gpuGlobalBuffer: shared.gpuGlobalBuffer,
        byteOffset,
        atomicOffset,
        globalOffset
    });
}

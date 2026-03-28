import { HypercubeManifest } from './types';
import { GpuEngine } from './GpuEngine';
import { GpuCoreFactory } from './GpuCoreFactory';
import { ManifestValidator } from './builders/ManifestValidator';

/**
 * The "Blessed Path" for creating a Hypercube GPU Simulation.
 * Validates the manifest and builds the engine in one call.
 * 
 * @param manifest The complete simulation manifest (engine + config).
 * @param mockDevice Optional GPUDevice for testing/headless environments.
 * @returns A fully initialized GpuEngine.
 */
export async function createSimulation<TParams = any, TFaces = any>(
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

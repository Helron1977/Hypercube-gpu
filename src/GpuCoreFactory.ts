import { HypercubeConfig, EngineDescriptor } from './types';
import { VirtualGrid } from './topology/VirtualGrid';
import { MasterBuffer } from './memory/MasterBuffer';
import { GpuDispatcher } from './dispatchers/GpuDispatcher';
import { ParityManager } from './ParityManager';
import { GpuEngine } from './GpuEngine';
import { HypercubeGPUContext } from './gpu/HypercubeGPUContext';

/**
 * Factory for creating GpuEngine instances from descriptors.
 */
export class GpuCoreFactory {
    public async build<TParams = any, TFaces = any>(config: HypercubeConfig<TParams>, engine: EngineDescriptor<TFaces>, mockDevice?: any): Promise<GpuEngine<TParams, TFaces>> {
        if (mockDevice) {
            await HypercubeGPUContext.init(mockDevice);
        } else if (!HypercubeGPUContext.isInitialized) {
            await HypercubeGPUContext.init();
        }

        const vGrid = new VirtualGrid(config, engine);
        const buffer = new MasterBuffer(vGrid, mockDevice);
        const parityManager = new ParityManager(vGrid.dataContract);
        const dispatcher = new GpuDispatcher(vGrid, buffer, parityManager, mockDevice);

        return new GpuEngine(vGrid, buffer, dispatcher, parityManager);
    }

    /**
     * Minimal helper to load a manifest from a URL.
     */
    public async loadManifest(url: string): Promise<any> {
        const res = await fetch(url);
        return res.json();
    }
}

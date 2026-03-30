import { HypercubeGPUContext } from '../gpu/HypercubeGPUContext';
import { WgslHeaderGenerator, WgslHeaderResult } from './WgslHeaderGenerator';

export interface PipelineMetadata {
    pipeline: GPUComputePipeline;
    usesAtomics: boolean;
    usesGlobals: boolean;
}

/**
 * Handles WebGPU Compute Pipeline caching with robust hashing.
 * v6.0: Enhanced with pre-calculated resource metadata (Zero-Stall).
 */
export class PipelineCache {
    private pipelines: Map<string, PipelineMetadata> = new Map();
    private currentDevice?: GPUDevice;

    constructor() {}

    /**
     * Retrieves or creates a compute pipeline for the given WGSL source.
     */
    public async getPipeline(
        device: GPUDevice, 
        ruleType: string, 
        source: string, 
        getWgslHeader: (type: string, source: string) => WgslHeaderResult
    ): Promise<PipelineMetadata> {
        // Context Loss / Device Change Protection
        if (this.currentDevice !== device) {
            this.currentDevice = device;
            this.clear();
        }

        const headerResult = getWgslHeader(ruleType, source);
        const combinedSource = headerResult.code + "\n" + source;
        
        // Robust Hashing (Fixes substring collision bug)
        const hash = this.computeHash(combinedSource);
        const cacheKey = `${ruleType}_${hash}`;

        let cached = this.pipelines.get(cacheKey);
        if (cached) return cached;

        const p = await HypercubeGPUContext.createComputePipelineAsync(combinedSource, `Kernel_${ruleType}`);
        
        // PRE-CALCULATE METADATA (v6.0 Alpha Zero-Stall)
        // Now using the DEFINITIVE flags from the header generator
        const metadata: PipelineMetadata = {
            pipeline: p,
            usesAtomics: headerResult.usesAtomics,
            usesGlobals: headerResult.usesGlobals
        };

        this.pipelines.set(cacheKey, metadata);
        return metadata;
    }

    private computeHash(str: string): number {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash |= 0; // Convert to 32bit integer
        }
        return hash;
    }

    public clear(): void {
        this.pipelines.clear();
    }
}

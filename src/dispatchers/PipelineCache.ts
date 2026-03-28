import { HypercubeGPUContext } from '../gpu/HypercubeGPUContext';

/**
 * Handles WebGPU Compute Pipeline caching with robust hashing.
 */
export class PipelineCache {
    private pipelines: Map<string, GPUComputePipeline> = new Map();
    private currentDevice?: GPUDevice;

    constructor() {}

    /**
     * Retrieves or creates a compute pipeline for the given WGSL source.
     */
    public async getPipeline(
        device: GPUDevice, 
        ruleType: string, 
        source: string, 
        getWgslHeader: (type: string) => string
    ): Promise<GPUComputePipeline> {
        // Context Loss / Device Change Protection
        if (this.currentDevice !== device) {
            this.currentDevice = device;
            this.clear();
        }

        const header = getWgslHeader(ruleType);
        const combinedSource = header + source;
        
        // Robust Hashing (Fixes substring collision bug)
        const hash = this.computeHash(combinedSource);
        const cacheKey = `${ruleType}_${hash}`;

        let p = this.pipelines.get(cacheKey);
        if (p) return p;

        p = await HypercubeGPUContext.createComputePipelineAsync(combinedSource, `Kernel_${ruleType}`);
        this.pipelines.set(cacheKey, p);
        return p;
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

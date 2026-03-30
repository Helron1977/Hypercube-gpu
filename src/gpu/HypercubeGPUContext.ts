/**
 * Global WebGPU Context for Hypercube Neo GPU Core.
 * Includes a self-healing "Auto-Mock" for headless unit testing.
 */

function ensureGlobalEnums() {
    if (typeof globalThis !== 'undefined') {
        const g = globalThis as any;
        if (!g.GPUBufferUsage) {
            g.GPUBufferUsage = {
                MAP_READ: 0x01, MAP_WRITE: 0x02, COPY_SRC: 0x04, COPY_DST: 0x08,
                INDEX: 0x10, VERTEX: 0x20, UNIFORM: 0x40, STORAGE: 0x80,
                INDIRECT: 0x100, QUERY_RESOLVE: 0x200
            };
        }
        if (!g.GPUMapMode) {
            g.GPUMapMode = { READ: 0x01, WRITE: 0x02 };
        }
        if (!g.GPUShaderStage) {
            g.GPUShaderStage = { VERTEX: 0x01, FRAGMENT: 0x02, COMPUTE: 0x04 };
        }
    }
}

// Execute immediately upon module load
ensureGlobalEnums();

export class HypercubeGPUContext {
    private static readonly STATE_KEY = Symbol.for('HYPERCUBE_GPU_CONTEXT_STATE');

    private static get _state(): { device: GPUDevice | null; adapter: GPUAdapter | null; isMock: boolean } {
        const g = globalThis as any;
        if (!g[this.STATE_KEY]) {
            g[this.STATE_KEY] = { device: null, adapter: null, isMock: false };
        }
        return g[this.STATE_KEY];
    }

    public static get isMock(): boolean {
        return this._state.isMock;
    }

    public static get device(): GPUDevice {
        const d = this._state.device;
        if (!d) {
            if (typeof process !== 'undefined' && process.env.NODE_ENV === 'test') {
                this.initAutoMock();
                return this._state.device!;
            }
            throw new Error("[HypercubeGPUContext] WebGPU device not initialized. Call HypercubeGPUContext.init() first.");
        }
        return d;
    }

    private static initAutoMock() {
        ensureGlobalEnums();
        const mockDevice = {
            createShaderModule: () => ({ label: 'mock-shader' }),
            createComputePipeline: () => ({ getBindGroupLayout: () => ({}), label: 'mock-pipeline' }),
            createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}), label: 'mock-pipeline' }),
            createBuffer: (desc: any) => ({ 
                destroy: () => {}, size: desc.size, 
                getMappedRange: () => new ArrayBuffer(desc.size),
                unmap: () => {}, mapAsync: () => Promise.resolve()
            }),
            createBindGroup: () => ({ label: 'mock-bind-group' }),
            queue: { 
                writeBuffer: () => {}, 
                submit: () => {},
                onSubmittedWorkDone: () => Promise.resolve()
            },
            limits: { minUniformBufferOffsetAlignment: 256 },
            createCommandEncoder: () => ({
                beginComputePass: () => ({ 
                    setPipeline: () => {}, setBindGroup: () => {}, 
                    setWorkgroupCount: () => {}, dispatchWorkgroups: () => {}, 
                    end: () => {} 
                }),
                copyBufferToBuffer: () => {}, finish: () => ({ label: 'mock-command-buffer' })
            })
        } as unknown as GPUDevice;
        this._state.device = mockDevice;
        this._state.isMock = true;
        (globalThis as any).__HYPERCUBE_IS_MOCK__ = true;
    }

    public static setDevice(device: GPUDevice): void {
        this._state.device = device;
        if (device && !device.pushErrorScope) {
            this._state.isMock = true;
            (globalThis as any).__HYPERCUBE_IS_MOCK__ = true;
        }
    }

    public static get isInitialized(): boolean {
        if (typeof process !== 'undefined' && process.env.NODE_ENV === 'test') return true;
        return this._state.device !== null;
    }

    static async init(mockDevice?: any): Promise<boolean> {
        if (mockDevice) {
            this.setDevice(mockDevice);
            return true;
        }
        if (this._state.device) return true;
        
        if (typeof navigator === 'undefined' || !navigator.gpu) {
            if (typeof process !== 'undefined' && process.env.NODE_ENV === 'test') {
                this.initAutoMock();
                return true;
            }
            console.error("[HypercubeGPUContext] WebGPU not supported.");
            return false;
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            console.error("[HypercubeGPUContext] Failed to get GPUAdapter.");
            return false;
        }
        this._state.adapter = adapter;

        this._state.device = await adapter.requestDevice({
            label: 'Hypercube GPU Core Device',
            requiredLimits: { 
                maxStorageBufferBindingSize: 1_073_741_824, 
                maxBufferSize: 1_073_741_824 
            }
        });
        this._state.isMock = false;
        (globalThis as any).__HYPERCUBE_IS_MOCK__ = false;

        console.info("[HypercubeGPUContext] WebGPU Device initialized.");
        return true;
    }

    public static get uniformAlignment(): number {
        const d = this.device;
        return d.limits.minUniformBufferOffsetAlignment || 256;
    }

    public static alignToUniform(size: number): number {
        const alignment = this.uniformAlignment;
        return Math.ceil(size / alignment) * alignment;
    }

    static createComputePipeline(wgslSource: string, label = 'Compute Pipeline'): GPUComputePipeline {
        const shaderModule = this.device.createShaderModule({ code: wgslSource });
        return this.device.createComputePipeline({
            label, layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' }
        });
    }

    static async createComputePipelineAsync(wgslSource: string, label = 'Compute Pipeline'): Promise<GPUComputePipeline> {
        const shaderModule = this.device.createShaderModule({ code: wgslSource });
        return await this.device.createComputePipelineAsync({
            label, layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' }
        });
    }

    static createStorageBuffer(size: number, usage?: number): GPUBuffer {
        return this.device.createBuffer({
            size: Math.ceil(size / 4) * 4,
            usage: usage !== undefined ? usage : (0x80 | 0x08 | 0x04),
            label: 'Hypercube Storage Buffer'
        });
    }

    static async loadKernel(url: string): Promise<string> {
        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(response.statusText);
            }
            return await response.text();
        } catch (e: any) {
            throw new Error(`Failed to load kernel from URL: ${e.message}`);
        }
    }

    static destroy() {
        this._state.device?.destroy();
        this._state.device = null;
        this._state.adapter = null;
        this._state.isMock = false;
        (globalThis as any).__HYPERCUBE_IS_MOCK__ = false;
    }
}

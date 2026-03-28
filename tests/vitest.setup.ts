import { vi } from 'vitest';
import { HypercubeGPUContext } from '../src/gpu/HypercubeGPUContext';

// Global WebGPU Mocks for Headless Testing
(globalThis as any).GPUBufferUsage = {
    MAP_READ: 0x01, MAP_WRITE: 0x02, COPY_SRC: 0x04, COPY_DST: 0x08,
    INDEX: 0x10, VERTEX: 0x20, UNIFORM: 0x40, STORAGE: 0x80,
    INDIRECT: 0x100, QUERY_RESOLVE: 0x200
};
(globalThis as any).GPUMapMode = { READ: 0x01, WRITE: 0x02 };
(globalThis as any).GPUShaderStage = { VERTEX: 0x01, FRAGMENT: 0x02, COMPUTE: 0x04 };

const mockDevice = {
    createShaderModule: () => ({}),
    createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }), 
    createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
    createBuffer: (desc: { size: number }) => ({ 
        destroy: () => {}, 
        size: desc.size, 
        getMappedRange: () => new ArrayBuffer(desc.size), 
        unmap: () => {}, 
        mapAsync: () => Promise.resolve() 
    }),
    createBindGroup: () => ({}),
    queue: { writeBuffer: () => {}, submit: () => {} },
    limits: { minUniformBufferOffsetAlignment: 256 },
    createCommandEncoder: () => ({
        beginComputePass: () => ({ setPipeline: () => {}, setBindGroup: () => {}, setWorkgroupCount: () => {}, dispatchWorkgroups: () => {}, end: () => {} }),
        copyBufferToBuffer: () => {},
        finish: () => ({})
    })
} as unknown as GPUDevice;

vi.stubGlobal('navigator', {
    gpu: {
        requestAdapter: async () => ({
            requestDevice: async () => mockDevice
        })
    }
});

// Pre-initialize for all tests in this worker
HypercubeGPUContext.init(mockDevice);

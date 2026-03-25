import { vi } from 'vitest';
import { HypercubeGPUContext } from '../src/gpu/HypercubeGPUContext';

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

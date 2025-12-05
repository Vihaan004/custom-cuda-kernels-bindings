#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <iostream>

// ReLU Forward Kernel
__global__ void relu_forward_kernel(const float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// ReLU Backward Kernel
__global__ void relu_backward_kernel(const float *grad_output, const float *input, float *grad_input, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

// C-style wrapper functions for ctypes
extern "C"
{
    // Memory management
    float *allocate_gpu_memory(size_t size)
    {
        float *ptr;
        cudaMalloc(&ptr, size * sizeof(float));
        return ptr;
    }

    void free_gpu_memory(float *ptr)
    {
        cudaFree(ptr);
    }

    void copy_to_gpu(float *gpu_ptr, const float *cpu_ptr, size_t size)
    {
        cudaMemcpy(gpu_ptr, cpu_ptr, size * sizeof(float), cudaMemcpyHostToDevice);
    }

    void copy_from_gpu(float *cpu_ptr, const float *gpu_ptr, size_t size)
    {
        cudaMemcpy(cpu_ptr, gpu_ptr, size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // ReLU forward operation
    void relu_forward_wrapper(float *input, float *output, int size)
    {
        dim3 threads_per_block(256);
        dim3 number_of_blocks((size + threads_per_block.x - 1) / threads_per_block.x);

        relu_forward_kernel<<<number_of_blocks, threads_per_block>>>(
            input, output, size);

        cudaDeviceSynchronize();
    }

    // ReLU backward operation
    void relu_backward_wrapper(float *grad_output, float *input, float *grad_input, int size)
    {

        // implement your code here - done
        dim3 threads_per_block(256);
        dim3 number_of_blocks((size + threads_per_block.x - 1) / threads_per_block.x);

        relu_backward_kernel<<<number_of_blocks, threads_per_block>>>(
            grad_output, input, grad_input, size);
        
        cudaDeviceSynchronize();
    }

    // Utility functions
    void cuda_synchronize()
    {
        cudaDeviceSynchronize();
    }

    int get_cuda_device_count()
    {
        int count;
        cudaGetDeviceCount(&count);
        return count;
    }
}
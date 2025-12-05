#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <iostream>

// Linear Forward Kernel
__global__ void linear_forward_kernel(const float *input, const float *weight, const float *bias,
                                      float *output, int batch_size, int input_size, int output_size)
{
    int batch_idx = blockIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && out_idx < output_size)
    {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++)
        {
            sum += input[batch_idx * input_size + i] * weight[out_idx * input_size + i];
        }
        output[batch_idx * output_size + out_idx] = sum + bias[out_idx];
    }
}

// Linear Backward Kernel for input gradients
__global__ void linear_backward_input_kernel(const float *grad_output, const float *weight,
                                             float *grad_input, int batch_size, int input_size, int output_size)
{
    int batch_idx = blockIdx.y;
    int in_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && in_idx < input_size)
    {
        float sum = 0.0f;
        for (int i = 0; i < output_size; i++)
        {
            sum += grad_output[batch_idx * output_size + i] * weight[i * input_size + in_idx];
        }
        grad_input[batch_idx * input_size + in_idx] = sum;
    }
}

// Linear Backward Kernel for weight gradients
__global__ void linear_backward_weight_kernel(const float *grad_output, const float *input,
                                              float *grad_weight, int batch_size, int input_size, int output_size)
{
    int out_idx = blockIdx.y;
    int in_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx < output_size && in_idx < input_size)
    {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++)
        {
            sum += grad_output[b * output_size + out_idx] * input[b * input_size + in_idx];
        }
        grad_weight[out_idx * input_size + in_idx] = sum;
    }
}

// Linear Backward Kernel for bias gradients
__global__ void linear_backward_bias_kernel(const float *grad_output, float *grad_bias,
                                            int batch_size, int output_size)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx < output_size)
    {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++)
        {
            sum += grad_output[b * output_size + out_idx];
        }
        grad_bias[out_idx] = sum;
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

    // Linear forward operation
    void linear_forward_wrapper(float *input, float *weight, float *bias, float *output,
                                int batch_size, int input_size, int output_size)
    {
        // implement your code here - done
        dim3 threads_per_block(256);
        dim3 number_blocks((output_size + threads_per_block.x - 1) / threads_per_block.x, batch_size);
        linear_forward_kernel<<<number_blocks, threads_per_block>>>(
            input, weight, bias, output, batch_size, input_size, output_size);
        cudaDeviceSynchronize();
    }

    // Linear backward operation
    void linear_backward_wrapper(float *grad_output, float *input, float *weight,
                                 float *grad_input, float *grad_weight, float *grad_bias,
                                 int batch_size, int input_size, int output_size)
    {
        dim3 threads_per_block(256);

        // Input gradients
        dim3 input_blocks((input_size + threads_per_block.x - 1) / threads_per_block.x, batch_size);
        linear_backward_input_kernel<<<input_blocks, threads_per_block>>>(
            grad_output, weight, grad_input, batch_size, input_size, output_size);

        // Weight gradients
        dim3 weight_blocks((input_size + threads_per_block.x - 1) / threads_per_block.x, output_size);
        linear_backward_weight_kernel<<<weight_blocks, threads_per_block>>>(
            grad_output, input, grad_weight, batch_size, input_size, output_size);

        // Bias gradients
        dim3 bias_blocks((output_size + threads_per_block.x - 1) / threads_per_block.x);
        linear_backward_bias_kernel<<<bias_blocks, threads_per_block>>>(
            grad_output, grad_bias, batch_size, output_size);

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
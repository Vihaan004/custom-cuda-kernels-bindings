#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ReLU Forward Kernel
__global__ void relu_forward_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// ReLU Backward Kernel
__global__ void relu_backward_kernel(const float* grad_output, const float* input, float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Gradient is 1 if input > 0, otherwise 0
        grad_input[idx] = input[idx] > 0.0f ? grad_output[idx] : 0.0f;
    }
}

// Linear Forward Kernel
__global__ void linear_forward_kernel(const float* input, const float* weight, const float* bias, 
                                     float* output, int batch_size, int input_size, int output_size) {
    int batch_idx = blockIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && out_idx < output_size) {
        float sum = 0.0f;
        // Matrix multiplication: output[batch, out] = sum(input[batch, in] * weight[out, in])
        for (int i = 0; i < input_size; i++) {
            sum += input[batch_idx * input_size + i] * weight[out_idx * input_size + i];
        }
        // Add bias
        output[batch_idx * output_size + out_idx] = sum + bias[out_idx];
    }
}

// Linear Backward Kernel for input gradients
__global__ void linear_backward_input_kernel(const float* grad_output, const float* weight, 
                                           float* grad_input, int batch_size, int input_size, int output_size) {
    int batch_idx = blockIdx.y;
    int in_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && in_idx < input_size) {
        float sum = 0.0f;
        for (int i = 0; i < output_size; i++) {
            sum += grad_output[batch_idx * output_size + i] * weight[i * input_size + in_idx];
        }
        grad_input[batch_idx * input_size + in_idx] = sum;
    }
}

// Linear Backward Kernel for weight gradients
__global__ void linear_backward_weight_kernel(const float* grad_output, const float* input, 
                                            float* grad_weight, int batch_size, int input_size, int output_size) {
    int out_idx = blockIdx.y;
    int in_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_idx < output_size && in_idx < input_size) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            sum += grad_output[b * output_size + out_idx] * input[b * input_size + in_idx];
        }
        grad_weight[out_idx * input_size + in_idx] = sum;
    }
}

// Linear Backward Kernel for bias gradients
__global__ void linear_backward_bias_kernel(const float* grad_output, float* grad_bias, 
                                          int batch_size, int output_size) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_idx < output_size) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            sum += grad_output[b * output_size + out_idx];
        }
        grad_bias[out_idx] = sum;
    }
}

// ReLU Forward Function
torch::Tensor relu_forward(torch::Tensor input) {
    const auto size = input.numel();
    auto output = torch::empty_like(input);
    
    auto input_contig = input.contiguous();
    auto output_contig = output.contiguous();
    
    dim3 threads_per_block(256);
    dim3 number_of_blocks((size + threads_per_block.x - 1) / threads_per_block.x);
    
    relu_forward_kernel<<<number_of_blocks, threads_per_block>>>(
        input_contig.data_ptr<float>(), output_contig.data_ptr<float>(), size);
    
    cudaDeviceSynchronize();
    return output_contig;
}

// ReLU Backward Function
torch::Tensor relu_backward(torch::Tensor grad_output, torch::Tensor input) {
    const auto size = input.numel();
    auto grad_input = torch::empty_like(input);
    
    auto grad_output_contig = grad_output.contiguous();
    auto input_contig = input.contiguous();
    auto grad_input_contig = grad_input.contiguous();
    
    dim3 threads_per_block(256);
    dim3 number_of_blocks((size + threads_per_block.x - 1) / threads_per_block.x);
    
    relu_backward_kernel<<<number_of_blocks, threads_per_block>>>(
        grad_output_contig.data_ptr<float>(), 
        input_contig.data_ptr<float>(), 
        grad_input_contig.data_ptr<float>(), 
        size);
    
    cudaDeviceSynchronize();
    return grad_input_contig;
}

// Linear Forward Function
torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    const auto batch_size = input.size(0);
    const auto input_size = input.size(1);
    const auto output_size = weight.size(0);
    
    auto output = torch::empty({batch_size, output_size}, input.options());
    
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();
    auto output_contig = output.contiguous();
    
    dim3 threads_per_block(256);
    dim3 number_of_blocks((output_size + threads_per_block.x - 1) / threads_per_block.x, batch_size);
    
    linear_forward_kernel<<<number_of_blocks, threads_per_block>>>(
        input_contig.data_ptr<float>(), 
        weight_contig.data_ptr<float>(), 
        bias_contig.data_ptr<float>(),
        output_contig.data_ptr<float>(), 
        batch_size, 
        input_size, 
        output_size);
    
    cudaDeviceSynchronize();
    return output_contig;
}

// Linear Backward Function
std::vector<torch::Tensor> linear_backward(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight) {
    const auto batch_size = input.size(0);
    const auto input_size = input.size(1);
    const auto output_size = weight.size(0);
    
    auto grad_output_contig = grad_output.contiguous();
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    
    // Compute input gradients
    auto grad_input = torch::empty_like(input);
    auto grad_input_contig = grad_input.contiguous();
    
    dim3 threads_per_block(256);
    dim3 input_blocks((input_size + threads_per_block.x - 1) / threads_per_block.x, batch_size);
    
    linear_backward_input_kernel<<<input_blocks, threads_per_block>>>(
        grad_output_contig.data_ptr<float>(), weight_contig.data_ptr<float>(),
        grad_input_contig.data_ptr<float>(), batch_size, input_size, output_size);
    
    // Compute weight gradients
    auto grad_weight = torch::empty_like(weight);
    auto grad_weight_contig = grad_weight.contiguous();
    
    dim3 weight_blocks((input_size + threads_per_block.x - 1) / threads_per_block.x, output_size);
    
    linear_backward_weight_kernel<<<weight_blocks, threads_per_block>>>(
        grad_output_contig.data_ptr<float>(), input_contig.data_ptr<float>(),
        grad_weight_contig.data_ptr<float>(), batch_size, input_size, output_size);
    
    // Compute bias gradients
    auto grad_bias = torch::empty({output_size}, input.options());
    auto grad_bias_contig = grad_bias.contiguous();
    
    dim3 bias_blocks((output_size + threads_per_block.x - 1) / threads_per_block.x);
    
    linear_backward_bias_kernel<<<bias_blocks, threads_per_block>>>(
        grad_output_contig.data_ptr<float>(), grad_bias_contig.data_ptr<float>(),
        batch_size, output_size);
    
    cudaDeviceSynchronize();
    
    return {grad_input_contig, grad_weight_contig, grad_bias_contig};
}

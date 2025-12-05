import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.cpp_extension import load_inline
from torch.utils.data import DataLoader

# CUDA ReLU and Linear kernels implementation
cuda_source = """
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
"""

cpp_source = """
torch::Tensor relu_forward(torch::Tensor input);
torch::Tensor relu_backward(torch::Tensor grad_output, torch::Tensor input);
torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
std::vector<torch::Tensor> linear_backward(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight);
"""


build_dir = "./load_inline_cuda"
if not os.path.exists(build_dir):
    os.makedirs(build_dir)
    print(f"Created build directory: {build_dir}")

# Load the CUDA kernel as a PyTorch extension
cuda_kernels = load_inline(
    name="cuda_kernels_extension",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["relu_forward", "relu_backward", "linear_forward", "linear_backward"],
    with_cuda=True,
    extra_cuda_cflags=["-O2"],
    build_directory=build_dir,
)


# Custom Autograd Functions
class CustomReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return cuda_kernels.relu_forward(input)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return cuda_kernels.relu_backward(grad_output, input)


class CustomLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        return cuda_kernels.linear_forward(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = cuda_kernels.linear_backward(
            grad_output, input, weight
        )
        return grad_input, grad_weight, grad_bias


EPOCHS = 10

# Load MNIST train and test datasets
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transforms.ToTensor()
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transforms.ToTensor()
)


# Define neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Manual weight and bias initialization for custom CUDA kernels
        self.fc1_weight = nn.Parameter(torch.randn(128, 28 * 28) * 0.1)
        self.fc1_bias = nn.Parameter(torch.randn(128) * 0.1)
        self.fc2_weight = nn.Parameter(torch.randn(10, 128) * 0.1)
        self.fc2_bias = nn.Parameter(torch.randn(10) * 0.1)

    def forward(self, x):
        x = x.view(-1, 28 * 28)

        # Custom CUDA Linear layer 1
        print(
            f"Using custom CUDA Linear on tensor shape: {x.shape}, device: {x.device}"
        )
        x = CustomLinear.apply(x, self.fc1_weight, self.fc1_bias)

        # Custom CUDA ReLU
        print(f"Using custom CUDA ReLU on tensor shape: {x.shape}, device: {x.device}")
        x = CustomReLU.apply(x)

        # Custom CUDA Linear layer 2
        print(
            f"Using custom CUDA Linear on tensor shape: {x.shape}, device: {x.device}"
        )
        x = CustomLinear.apply(x, self.fc2_weight, self.fc2_bias)

        return x


# Initialize model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training loop
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
print(f"\n Starting MNIST training with custom CUDA kernels for {EPOCHS} epochs...")
print("=" * 60)

for epoch in range(EPOCHS):
    total_loss = 0
    num_batches = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 100 == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}, Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1} completed - Average Loss: {avg_loss:.4f}")
    print("-" * 60)

print("Training completed with custom CUDA kernels!")
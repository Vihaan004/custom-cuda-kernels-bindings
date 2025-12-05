# Custom CUDA Kernels: PyTorch Extension + ctypes Bindings

This repository showcases two complementary approaches to accelerating core tensor operations (Linear Forward and ReLU Backward) on GPU using custom CUDA kernels:

- PyTorch C++/CUDA Extension: Integrates CUDA kernels directly with `torch::Tensor` wrappers for end-to-end training on MNIST.
- Python ctypes Bindings: Builds a shared CUDA library and calls kernels from Python without relying on PyTorch’s extension mechanism.

## Highlights
- Linear Forward: Matrix multiply + bias ($Y = XW + b$), batched.
- ReLU Backward: Gradient propagation through ReLU ($\nabla X = \nabla Y \odot [X>0]$).
- GPU-first implementations with minimal glue code to keep the core logic visible.

## Project Layout

```
.
├── pytorch-custom-kernels/           # PyTorch extension + training
│   ├── training.py                   # MNIST training harness
│   └── extension_src/                # C++/CUDA sources (kernels + bindings)
├── ctypes-cuda-bindings/             # CUDA shared lib + ctypes wrappers
│   ├── training.py                   # MNIST training using ctypes-exposed kernels
│   └── src/                          # CUDA kernel sources
└── data/
    └── MNIST/
        └── raw/                      # Shared dataset files
```

## Core Files to Explore
- PyTorch path:
  - `pytorch-custom-kernels/extension_src/cuda.cu`: CUDA kernels (Linear Forward, ReLU Backward).
  - `pytorch-custom-kernels/extension_src/main.cpp`: C++ bindings with `torch::Tensor` wrappers.
  - `pytorch-custom-kernels/training.py`: Uses the extension during MNIST training.
- ctypes path:
  - `ctypes-cuda-bindings/src/linear_wrapper.cu`: Linear Forward kernel and wrapper.
  - `ctypes-cuda-bindings/src/relu_wrapper.cu`: ReLU Backward kernel and wrapper.
  - `ctypes-cuda-bindings/training.py`: Loads the shared library via `ctypes` and trains MNIST.

## Quick Start

Prerequisites: CUDA toolkit, Python 3.10+, PyTorch (for the extension approach), and MNIST data under `data/MNIST/raw/`.

- PyTorch Extension approach:
  ```bash
  cd pytorch-custom-kernels
  python3 training.py
  ```

- ctypes Bindings approach:
  ```bash
  cd ctypes-cuda-bindings
  # Ensure the CUDA sources are compiled to a shared library (e.g., libkernels.so)
  # Your training script will load the .so via ctypes
  python3 training.py
  ```

## Conceptual Overview
- Linear Forward:
  - Each output element computes a dot product over input features plus bias.
  - CUDA kernel maps threads to output matrix tiles for parallelism and coalesced memory access.
- ReLU Backward:
  - Gradient is passed where input > 0, zeroed otherwise.
  - CUDA kernel applies an element-wise mask to upstream gradients.

## Notes
- The dataset directory is shared across both implementations; adjust paths in training scripts if needed.
- File and folder names are neutral to position this work as portfolio-ready rather than course-specific.

import ctypes
import os
import struct
import subprocess
import sys
from typing import List, Tuple

import numpy as np


# MNIST Data Loading Functions
def load_mnist_images(filename: str) -> np.ndarray:
    """Load MNIST images from IDX file format"""
    with open(filename, "rb") as f:
        # Read magic number
        magic = struct.unpack(">I", f.read(4))[0]
        if magic != 2051:
            raise ValueError(f"Invalid magic number: {magic}")

        # Read number of images, rows, cols
        num_images = struct.unpack(">I", f.read(4))[0]
        rows = struct.unpack(">I", f.read(4))[0]
        cols = struct.unpack(">I", f.read(4))[0]

        # Read image data
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)

        # Normalize to [0, 1] and flatten
        images = images.astype(np.float32) / 255.0
        images = images.reshape(num_images, -1)

        return images


def load_mnist_labels(filename: str) -> np.ndarray:
    """Load MNIST labels from IDX file format"""
    with open(filename, "rb") as f:
        # Read magic number
        magic = struct.unpack(">I", f.read(4))[0]
        if magic != 2049:
            raise ValueError(f"Invalid magic number: {magic}")

        # Read number of labels
        num_labels = struct.unpack(">I", f.read(4))[0]

        # Read label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)

        # Validate that we read the expected number of labels
        if len(labels) != num_labels:
            raise ValueError(f"Expected {num_labels} labels, got {len(labels)}")

        return labels


def load_mnist_data(
    data_dir: str = "./data/MNIST/raw",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST train and test datasets"""
    train_images = load_mnist_images(os.path.join(data_dir, "train-images-idx3-ubyte"))
    train_labels = load_mnist_labels(os.path.join(data_dir, "train-labels-idx1-ubyte"))
    test_images = load_mnist_images(os.path.join(data_dir, "t10k-images-idx3-ubyte"))
    test_labels = load_mnist_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte"))

    return train_images, train_labels, test_images, test_labels


def create_batches(
    images: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool = True
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create batches from images and labels"""
    num_samples = len(images)
    indices = np.arange(num_samples)

    if shuffle:
        np.random.shuffle(indices)

    batches = []
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        batch_indices = indices[i:end_idx]

        batch_images = images[batch_indices]
        batch_labels = labels[batch_indices]

        batches.append((batch_images, batch_labels))

    return batches


def one_hot_encode(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Convert labels to one-hot encoding"""
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return one_hot


def softmax(x):
    """Compute softmax values for each row"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(y_pred, y_true):
    """Compute cross-entropy loss"""
    # Convert to one-hot if needed
    if y_true.ndim == 1:
        y_true = one_hot_encode(y_true)

    # Add small epsilon to avoid log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Compute loss
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    return loss


class CUDANeuralNetwork:
    def __init__(
        self, input_size=784, hidden_size=128, output_size=10, learning_rate=0.001
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Load the compiled libraries
        self.linear_lib = self._load_linear_library()
        self.relu_lib = self._load_relu_library()

        # Initialize weights and biases
        self._initialize_weights()

        # Allocate GPU memory
        self._allocate_gpu_memory()

    def _load_linear_library(self):
        """Load the compiled linear library using ctypes"""
        # Compile the linear wrapper if it doesn't exist
        if not os.path.exists("./linear_wrapper.so"):
            self._compile_linear_library()

        # Load the shared library
        lib = ctypes.CDLL("./linear_wrapper.so")

        # Define function signatures
        lib.allocate_gpu_memory.argtypes = [ctypes.c_size_t]
        lib.allocate_gpu_memory.restype = ctypes.c_void_p

        lib.free_gpu_memory.argtypes = [ctypes.c_void_p]
        lib.free_gpu_memory.restype = None

        # implement your code here - done

        lib.copy_to_gpu.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        lib.copy_to_gpu.restype = None

        lib.copy_from_gpu.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_size_t]
        lib.copy_from_gpu.restype = None

        lib.linear_forward_wrapper.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        lib.linear_forward_wrapper.restype = None

        lib.linear_backward_wrapper.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        lib.linear_backward_wrapper.restype = None

        return lib
        # pass

    def _load_relu_library(self):
        """Load the compiled ReLU library using ctypes"""
        # Compile the ReLU wrapper if it doesn't exist
        if not os.path.exists("./relu_wrapper.so"):
            self._compile_relu_library()

        # Load the shared library
        lib = ctypes.CDLL("./relu_wrapper.so")

        # Define function signatures
        lib.allocate_gpu_memory.argtypes = [ctypes.c_size_t]
        lib.allocate_gpu_memory.restype = ctypes.c_void_p

        lib.free_gpu_memory.argtypes = [ctypes.c_void_p]
        lib.free_gpu_memory.restype = None

        # implement your code here - done
        lib.relu_forward_wrapper.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.relu_forward_wrapper.restype = None

        lib.relu_backward_wrapper.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.relu_backward_wrapper.restype = None

        return lib

    def _compile_linear_library(self):
        """Compile the linear wrapper into a shared library"""
        print("Compiling linear CUDA wrapper...")
        cmd = [
            "nvcc",
            "-shared",
            "-o",
            "linear_wrapper.so",
            "linear_wrapper.cu",
            "-lcudart",
            "-lcuda",
            "-Xcompiler",
            "-fPIC",
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Linear wrapper compilation successful!")
        except subprocess.CalledProcessError as e:
            print(f"Linear wrapper compilation failed: {e}")
            print(f"Error output: {e.stderr}")
            sys.exit(1)

    def _compile_relu_library(self):
        """Compile the ReLU wrapper into a shared library"""
        print("Compiling ReLU CUDA wrapper...")
        cmd = [
            "nvcc",
            "-shared",
            "-o",
            "relu_wrapper.so",
            "relu_wrapper.cu",
            "-lcudart",
            "-lcuda",
            "-Xcompiler",
            "-fPIC",
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("ReLU wrapper compilation successful!")
        except subprocess.CalledProcessError as e:
            print(f"ReLU wrapper compilation failed: {e}")
            print(f"Error output: {e.stderr}")
            sys.exit(1)

    def _initialize_weights(self):
        """Initialize weights and biases with random values"""
        # Xavier initialization
        self.W1 = (
            np.random.randn(self.hidden_size, self.input_size).astype(np.float32) * 0.1
        )
        self.b1 = np.random.randn(self.hidden_size).astype(np.float32) * 0.1
        self.W2 = (
            np.random.randn(self.output_size, self.hidden_size).astype(np.float32) * 0.1
        )
        self.b2 = np.random.randn(self.output_size).astype(np.float32) * 0.1

    def _allocate_gpu_memory(self):
        """Allocate GPU memory for all tensors"""
        # Calculate sizes
        self.max_batch_size = 64  # We'll use this as our maximum batch size

        # Allocate memory for inputs/outputs
        self.gpu_input = self.linear_lib.allocate_gpu_memory(
            self.max_batch_size * self.input_size
        )
        self.gpu_hidden = self.linear_lib.allocate_gpu_memory(
            self.max_batch_size * self.hidden_size
        )
        # self.gpu_hidden_pre_relu = self.linear_lib.allocate_gpu_memory(
        #     self.max_batch_size * self.hidden_size
        # )
        self.gpu_output = self.linear_lib.allocate_gpu_memory(
            self.max_batch_size * self.output_size
        )

        # Allocate memory for weights and biases
        self.gpu_W1 = self.linear_lib.allocate_gpu_memory(
            self.hidden_size * self.input_size
        )
        self.gpu_b1 = self.linear_lib.allocate_gpu_memory(self.hidden_size)
        self.gpu_W2 = self.linear_lib.allocate_gpu_memory(
            self.output_size * self.hidden_size
        )
        self.gpu_b2 = self.linear_lib.allocate_gpu_memory(self.output_size)

        # Allocate memory for gradients
        self.gpu_grad_hidden = self.linear_lib.allocate_gpu_memory(
            self.max_batch_size * self.hidden_size
        )
        self.gpu_grad_input = self.linear_lib.allocate_gpu_memory(
            self.max_batch_size * self.input_size
        )
        self.gpu_grad_W1 = self.linear_lib.allocate_gpu_memory(
            self.hidden_size * self.input_size
        )
        self.gpu_grad_b1 = self.linear_lib.allocate_gpu_memory(self.hidden_size)
        self.gpu_grad_W2 = self.linear_lib.allocate_gpu_memory(
            self.output_size * self.hidden_size
        )
        self.gpu_grad_b2 = self.linear_lib.allocate_gpu_memory(self.output_size)

        # Copy initial weights to GPU
        self._copy_weights_to_gpu()

    def _copy_weights_to_gpu(self):
        """Copy weights and biases to GPU memory"""
        # Convert to ctypes arrays
        W1_ptr = self.W1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b1_ptr = self.b1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        W2_ptr = self.W2.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b2_ptr = self.b2.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Copy to GPU
        self.linear_lib.copy_to_gpu(
            self.gpu_W1, W1_ptr, self.hidden_size * self.input_size
        )
        self.linear_lib.copy_to_gpu(self.gpu_b1, b1_ptr, self.hidden_size)
        self.linear_lib.copy_to_gpu(
            self.gpu_W2, W2_ptr, self.output_size * self.hidden_size
        )
        self.linear_lib.copy_to_gpu(self.gpu_b2, b2_ptr, self.output_size)

    def _copy_weights_from_gpu(self):
        """Copy weights and biases from GPU memory"""
        # Allocate CPU memory
        W1_cpu = np.zeros((self.hidden_size, self.input_size), dtype=np.float32)
        b1_cpu = np.zeros(self.hidden_size, dtype=np.float32)
        W2_cpu = np.zeros((self.output_size, self.hidden_size), dtype=np.float32)
        b2_cpu = np.zeros(self.output_size, dtype=np.float32)

        # Convert to ctypes arrays
        W1_ptr = W1_cpu.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b1_ptr = b1_cpu.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        W2_ptr = W2_cpu.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b2_ptr = b2_cpu.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Copy from GPU
        self.linear_lib.copy_from_gpu(
            W1_ptr, self.gpu_W1, self.hidden_size * self.input_size
        )
        self.linear_lib.copy_from_gpu(b1_ptr, self.gpu_b1, self.hidden_size)
        self.linear_lib.copy_from_gpu(
            W2_ptr, self.gpu_W2, self.output_size * self.hidden_size
        )
        self.linear_lib.copy_from_gpu(b2_ptr, self.gpu_b2, self.output_size)

        # Update weights
        self.W1 = W1_cpu
        self.b1 = b1_cpu
        self.W2 = W2_cpu
        self.b2 = b2_cpu

    def forward(self, X):
        """Forward pass through the network"""
        batch_size = X.shape[0]

        # Copy input to GPU
        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.linear_lib.copy_to_gpu(self.gpu_input, X_ptr, batch_size * self.input_size)

        # First linear layer: X @ W1.T + b1
        # implement your code here - done
        self.linear_lib.linear_forward_wrapper(
            self.gpu_input,
            self.gpu_W1,
            self.gpu_b1,
            self.gpu_hidden,
            batch_size,
            self.input_size,
            self.hidden_size,
        )

        # ReLU activation
        self.relu_lib.relu_forward_wrapper(
            self.gpu_hidden, self.gpu_hidden, batch_size * self.hidden_size
        )

        # Second linear layer: hidden @ W2.T + b2
        # implement your code here - done
        self.linear_lib.linear_forward_wrapper(
            self.gpu_hidden,
            self.gpu_W2,
            self.gpu_b2,
            self.gpu_output,
            batch_size,
            self.hidden_size,
            self.output_size,
        )

        # Copy output back to CPU
        output = np.zeros((batch_size, self.output_size), dtype=np.float32)
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.linear_lib.copy_from_gpu(
            output_ptr, self.gpu_output, batch_size * self.output_size
        )

        return output

    def backward(self, X, y, output):
        """Backward pass to compute gradients"""
        batch_size = X.shape[0]

        # Compute output gradients (softmax + cross-entropy derivative)
        # For simplicity, we'll use a basic gradient computation
        grad_output = output - y  # This is a simplified version

        # Copy gradients to GPU
        grad_output_ptr = grad_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.linear_lib.copy_to_gpu(
            self.gpu_output, grad_output_ptr, batch_size * self.output_size
        )

        # Backward through second linear layer
        self.linear_lib.linear_backward_wrapper(
            self.gpu_output,
            self.gpu_hidden,
            self.gpu_W2,
            self.gpu_grad_hidden,
            self.gpu_grad_W2,
            self.gpu_grad_b2,
            batch_size,
            self.hidden_size,
            self.output_size,
        )

        # Backward through ReLU
        # implement your code here - done
        self.relu_lib.relu_backward_wrapper(
            self.gpu_grad_hidden,
            self.gpu_hidden,
            self.gpu_grad_hidden,
            batch_size * self.hidden_size
        )

        # Backward through first linear layer
        self.linear_lib.linear_backward_wrapper(
            self.gpu_grad_hidden,
            self.gpu_input,
            self.gpu_W1,
            self.gpu_grad_input,
            self.gpu_grad_W1,
            self.gpu_grad_b1,
            batch_size,
            self.input_size,
            self.hidden_size,
        )

        # Update weights (simplified gradient descent)
        self._update_weights()

    def _update_weights(self):
        """Update weights using computed gradients"""
        # Copy gradients from GPU
        grad_W1 = np.zeros((self.hidden_size, self.input_size), dtype=np.float32)
        grad_b1 = np.zeros(self.hidden_size, dtype=np.float32)
        grad_W2 = np.zeros((self.output_size, self.hidden_size), dtype=np.float32)
        grad_b2 = np.zeros(self.output_size, dtype=np.float32)

        # Convert to ctypes arrays
        grad_W1_ptr = grad_W1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        grad_b1_ptr = grad_b1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        grad_W2_ptr = grad_W2.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        grad_b2_ptr = grad_b2.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Copy from GPU
        self.linear_lib.copy_from_gpu(
            grad_W1_ptr, self.gpu_grad_W1, self.hidden_size * self.input_size
        )
        self.linear_lib.copy_from_gpu(grad_b1_ptr, self.gpu_grad_b1, self.hidden_size)
        self.linear_lib.copy_from_gpu(
            grad_W2_ptr, self.gpu_grad_W2, self.output_size * self.hidden_size
        )
        self.linear_lib.copy_from_gpu(grad_b2_ptr, self.gpu_grad_b2, self.output_size)

        # Update weights
        self.W1 -= self.learning_rate * grad_W1
        self.b1 -= self.learning_rate * grad_b1
        self.W2 -= self.learning_rate * grad_W2
        self.b2 -= self.learning_rate * grad_b2

        # Copy updated weights back to GPU
        self._copy_weights_to_gpu()

    def predict(self, X):
        """Make predictions on input data"""
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def __del__(self):
        """Clean up GPU memory"""
        if hasattr(self, "linear_lib"):
            self.linear_lib.free_gpu_memory(self.gpu_input)
            self.linear_lib.free_gpu_memory(self.gpu_hidden)
            # self.linear_lib.free_gpu_memory(self.gpu_hidden_pre_relu)
            self.linear_lib.free_gpu_memory(self.gpu_output)
            self.linear_lib.free_gpu_memory(self.gpu_W1)
            self.linear_lib.free_gpu_memory(self.gpu_b1)
            self.linear_lib.free_gpu_memory(self.gpu_W2)
            self.linear_lib.free_gpu_memory(self.gpu_b2)
            self.linear_lib.free_gpu_memory(self.gpu_grad_hidden)
            self.linear_lib.free_gpu_memory(self.gpu_grad_input)
            self.linear_lib.free_gpu_memory(self.gpu_grad_W1)
            self.linear_lib.free_gpu_memory(self.gpu_grad_b1)
            self.linear_lib.free_gpu_memory(self.gpu_grad_W2)
            self.linear_lib.free_gpu_memory(self.gpu_grad_b2)


def train_model():
    """Main training function"""
    print("Loading MNIST data...")
    train_images, train_labels, test_images, test_labels = load_mnist_data()

    print(f"Training set: {train_images.shape[0]} samples")
    print(f"Test set: {test_images.shape[0]} samples")

    # Initialize model
    model = CUDANeuralNetwork(learning_rate=0.001)

    # Training parameters
    epochs = 5
    batch_size = 64

    print(f"\n Starting MNIST training with ctypes CUDA wrapper for {epochs} epochs...")
    print("=" * 60)

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        # Create batches
        batches = create_batches(train_images, train_labels, batch_size, shuffle=True)

        for batch_idx, (batch_images, batch_labels) in enumerate(batches):
            # Convert labels to one-hot
            y_true = one_hot_encode(batch_labels)

            # Forward pass
            output = model.forward(batch_images)
            output_softmax = softmax(output)

            # Compute loss
            loss = cross_entropy_loss(output_softmax, batch_labels)
            total_loss += loss
            num_batches += 1

            # Backward pass
            model.backward(batch_images, y_true, output_softmax)

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss:.4f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} completed - Average Loss: {avg_loss:.4f}")
        print("-" * 60)

    # Test the model
    print("\nTesting model...")
    test_batches = create_batches(test_images, test_labels, batch_size, shuffle=False)

    correct = 0
    total = 0

    for batch_images, batch_labels in test_batches:
        predictions = model.predict(batch_images)
        correct += np.sum(predictions == batch_labels)
        total += len(batch_labels)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")

    print("Training completed with ctypes CUDA wrapper!")


if __name__ == "__main__":
    train_model()

#include <torch/extension.h>

torch::Tensor relu_forward(torch::Tensor input);
torch::Tensor relu_backward(torch::Tensor grad_output, torch::Tensor input);
torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
std::vector<torch::Tensor> linear_backward(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("relu_forward", torch::wrap_pybind_function(relu_forward), "relu_forward");
m.def("relu_backward", torch::wrap_pybind_function(relu_backward), "relu_backward");
m.def("linear_forward", torch::wrap_pybind_function(linear_forward), "linear_forward");
m.def("linear_backward", torch::wrap_pybind_function(linear_backward), "linear_backward");
}
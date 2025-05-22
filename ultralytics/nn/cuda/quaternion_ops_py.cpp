#include <torch/extension.h>
#include <vector>

// Forward declarations for CUDA functions
torch::Tensor qconv_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight_r,
    torch::Tensor weight_i,
    torch::Tensor weight_j,
    torch::Tensor weight_k,
    torch::Tensor bias_r,
    torch::Tensor bias_i,
    torch::Tensor bias_j,
    torch::Tensor bias_k,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups);

torch::Tensor iqbn_forward_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps);

torch::Tensor qconv_forward(
    torch::Tensor input,
    torch::Tensor weight_r,
    torch::Tensor weight_i,
    torch::Tensor weight_j,
    torch::Tensor weight_k,
    c10::optional<torch::Tensor> bias_r_opt,     // MODIFIED
    c10::optional<torch::Tensor> bias_i_opt,     // MODIFIED
    c10::optional<torch::Tensor> bias_j_opt,     // MODIFIED
    c10::optional<torch::Tensor> bias_k_opt,     // MODIFIED
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups) {
        
    // Convert optional to torch::Tensor. If Python passed None, .has_value() is false,
    // and torch::Tensor() creates an undefined tensor.
    torch::Tensor bias_r = bias_r_opt.has_value() ? *bias_r_opt : torch::Tensor();
    torch::Tensor bias_i = bias_i_opt.has_value() ? *bias_i_opt : torch::Tensor();
    torch::Tensor bias_j = bias_j_opt.has_value() ? *bias_j_opt : torch::Tensor();
    torch::Tensor bias_k = bias_k_opt.has_value() ? *bias_k_opt : torch::Tensor();

    // Input validation
    TORCH_CHECK(input.device().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight_r.device().is_cuda(), "weight_r must be a CUDA tensor");
    TORCH_CHECK(weight_i.device().is_cuda(), "weight_i must be a CUDA tensor");
    TORCH_CHECK(weight_j.device().is_cuda(), "weight_j must be a CUDA tensor");
    TORCH_CHECK(weight_k.device().is_cuda(), "weight_k must be a CUDA tensor");

    // Check bias tensors only if they are defined (not None in Python)
    bool has_bias = bias_r.defined();
    if (has_bias) {
         TORCH_CHECK(bias_i.defined() && bias_j.defined() && bias_k.defined(),
                    "If bias_r is provided, bias_i, bias_j, and bias_k must also be provided.");
         TORCH_CHECK(bias_r.device().is_cuda(), "bias_r must be a CUDA tensor");
         TORCH_CHECK(bias_i.device().is_cuda(), "bias_i must be a CUDA tensor");
         TORCH_CHECK(bias_j.device().is_cuda(), "bias_j must be a CUDA tensor");
         TORCH_CHECK(bias_k.device().is_cuda(), "bias_k must be a CUDA tensor");
    } else {
        // Ensure other biases are also undefined if bias_r isn't
         TORCH_CHECK(!bias_i.defined() && !bias_j.defined() && !bias_k.defined(),
                    "If bias_r is None, bias_i, bias_j, and bias_k must also be None.");
    }
    // Pass dilation argument which was missing before
    return qconv_forward_cuda(input, weight_r, weight_i, weight_j, weight_k,
                              bias_r, bias_i, bias_j, bias_k,
                              stride, padding, dilation, groups);
}

torch::Tensor iqbn_forward(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps) {

    // Input validation
    TORCH_CHECK(input.device().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(gamma.device().is_cuda(), "Gamma must be a CUDA tensor");
    TORCH_CHECK(beta.device().is_cuda(), "Beta must be a CUDA tensor");
    TORCH_CHECK(running_mean.device().is_cuda(), "Running mean must be a CUDA tensor");
    TORCH_CHECK(running_var.device().is_cuda(), "Running variance must be a CUDA tensor");

    return iqbn_forward_cuda(input, gamma, beta, running_mean, running_var, eps);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qconv_forward", &qconv_forward, "Quaternion Convolution Forward with Hamilton Product (CUDA)"
        , py::arg("input")
        , py::arg("weight_r")
        , py::arg("weight_i")
        , py::arg("weight_j")
        , py::arg("weight_k")
        , py::arg("bias_r") // With c10::optional, pybind11 usually handles None conversion automatically
        , py::arg("bias_i")
        , py::arg("bias_j")
        , py::arg("bias_k")
        , py::arg("stride")
        , py::arg("padding")
        , py::arg("dilation")
        , py::arg("groups")
    );
    m.def("iqbn_forward", &iqbn_forward, "Independent Quaternion BatchNorm Forward (CUDA)");
}

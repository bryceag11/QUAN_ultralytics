// quaternion_ops_head.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t, typename acc_scalar_t = float>
__global__ void qmix_forward_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int B,
    const int C,
    const int H,
    const int W,
    const int out_components) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = B * C * H * W;
    
    if (tid >= total_elements) return;
    
    // Decompose indices
    const int w = tid % W;
    const int h = (tid / W) % H;
    const int c = (tid / (W * H)) % C;
    const int b = tid / (W * H * C);
    
    // Input indices for quaternion components
    const int base_idx = tid * 4;
    const scalar_t xr = input[base_idx + 0];
    const scalar_t xi = input[base_idx + 1];
    const scalar_t xj = input[base_idx + 2];
    const scalar_t xk = input[base_idx + 3];
    
    // Apply separable Hamilton mixing
    // Forward signs from your specification:
    const acc_scalar_t yr = xr + xi + xj + xk;
    const acc_scalar_t yi = -xi + xr + xk - xj;
    const acc_scalar_t yj = -xj - xk + xr + xi;
    const acc_scalar_t yk = -xk + xj - xi + xr;
    
    // Output based on out_components
    if (out_components == 1) {
        // Only output real component
        output[tid] = static_cast<scalar_t>(yr);
    } else if (out_components == 4) {
        // Output all components
        const int out_base = tid * 4;
        output[out_base + 0] = static_cast<scalar_t>(yr);
        output[out_base + 1] = static_cast<scalar_t>(yi);
        output[out_base + 2] = static_cast<scalar_t>(yj);
        output[out_base + 3] = static_cast<scalar_t>(yk);
    }
}

template <typename scalar_t, typename acc_scalar_t = float>
__global__ void qmix_backward_kernel(
    scalar_t* __restrict__ grad_input,
    const scalar_t* __restrict__ grad_output,
    const int B,
    const int C,
    const int H,
    const int W,
    const int out_components) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = B * C * H * W;
    
    if (tid >= total_elements) return;
    
    // Get gradients from output
    acc_scalar_t grad_r, grad_i, grad_j, grad_k;
    
    if (out_components == 1) {
        // Only real component gradient
        grad_r = static_cast<acc_scalar_t>(grad_output[tid]);
        grad_i = 0;
        grad_j = 0;
        grad_k = 0;
    } else if (out_components == 4) {
        // All component gradients
        const int out_base = tid * 4;
        grad_r = static_cast<acc_scalar_t>(grad_output[out_base + 0]);
        grad_i = static_cast<acc_scalar_t>(grad_output[out_base + 1]);
        grad_j = static_cast<acc_scalar_t>(grad_output[out_base + 2]);
        grad_k = static_cast<acc_scalar_t>(grad_output[out_base + 3]);
    }
    
    // Apply backward mixing using provided signs
    const int base_idx = tid * 4;
    grad_input[base_idx + 0] = static_cast<scalar_t>(grad_r + grad_i + grad_j + grad_k);
    grad_input[base_idx + 1] = static_cast<scalar_t>(- grad_i + grad_r - grad_k + grad_j);
    grad_input[base_idx + 2] = static_cast<scalar_t>(- grad_j + grad_k + grad_r - grad_i);
    grad_input[base_idx + 3] = static_cast<scalar_t>(- grad_k - grad_j + grad_i + grad_r);
}

torch::Tensor qmix_forward_cuda(
    torch::Tensor input,        // [B, C, H, W, 4]
    int out_components) {       // 1 or 4
    
    TORCH_CHECK(input.dim() == 5, "qmix_forward_cuda expects 5D input");
    TORCH_CHECK(input.size(4) == 4, "Input quaternion dimension must be 4");
    TORCH_CHECK(out_components == 1 || out_components == 4, "out_components must be 1 or 4");
    
    const auto B = input.size(0);
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);
    
    // Create output tensor
    torch::Tensor output;
    if (out_components == 1) {
        output = torch::empty({B, C, H, W}, input.options());
    } else {
        output = torch::empty({B, C, H, W, 4}, input.options());
    }
    
    const int threads = 512;
    const int total_elements = B * C * H * W;
    const int blocks = (total_elements + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "qmix_forward_cuda", ([&] {
        qmix_forward_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            B, C, H, W, out_components
        );
    }));
    
    return output;
}

torch::Tensor qmix_backward_cuda(
    torch::Tensor grad_output,
    int B, int C, int H, int W,
    int out_components) {
    
    TORCH_CHECK(out_components == 1 || out_components == 4, "out_components must be 1 or 4");
    
    // Create grad_input tensor
    auto grad_input = torch::zeros({B, C, H, W, 4}, grad_output.options());
    
    const int threads = 512;
    const int total_elements = B * C * H * W;
    const int blocks = (total_elements + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "qmix_backward_cuda", ([&] {
        qmix_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_input.data_ptr<scalar_t>(),
            grad_output.data_ptr<scalar_t>(),
            B, C, H, W, out_components
        );
    }));
    
    return grad_input;
}
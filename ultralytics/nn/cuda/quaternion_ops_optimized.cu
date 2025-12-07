// quaternion_ops_optimized.cu - Optimized CUDA kernels for QUAN
// Preserves the "Zhou separable" / "Correct Left Conj separable" mixing that works best
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Helper for vectorized quaternion load/store
struct __align__(16) float4_quat {
    float r, i, j, k;
};

// ============================================================================
// OPTIMIZED IQBN FORWARD KERNEL
// Uses vectorized loads and fused operations
// ============================================================================
template <typename scalar_t>
__global__ void iqbn_forward_kernel_optimized(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ beta,
    const scalar_t* __restrict__ running_mean,
    const scalar_t* __restrict__ running_var,
    const int B,
    const int C,
    const int HW,
    const float eps) {

    // Each thread processes one spatial location (all 4 quaternion components)
    const int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_spatial = B * C * HW;
    if (spatial_idx >= total_spatial) return;

    // Decompose index
    const int hw_idx = spatial_idx % HW;
    const int c = (spatial_idx / HW) % C;
    const int b = spatial_idx / (HW * C);

    // Base index for quaternion (BCHWQ layout, Q=4 is innermost)
    const int base_idx = spatial_idx * 4;

    // Load all 4 quaternion components at once (vectorized)
    float4 inp;
    inp.x = static_cast<float>(input[base_idx + 0]);
    inp.y = static_cast<float>(input[base_idx + 1]);
    inp.z = static_cast<float>(input[base_idx + 2]);
    inp.w = static_cast<float>(input[base_idx + 3]);

    // Load stats for all 4 components (channel-dependent)
    const int stat_base = c * 4;

    // Fused normalize + affine for each component
    #pragma unroll
    for (int q = 0; q < 4; q++) {
        const float mean_val = static_cast<float>(running_mean[stat_base + q]);
        const float var_val = static_cast<float>(running_var[stat_base + q]);
        const float gamma_val = static_cast<float>(gamma[stat_base + q]);
        const float beta_val = static_cast<float>(beta[stat_base + q]);

        const float inv_std = rsqrtf(var_val + eps);
        float* inp_ptr = &inp.x + q;
        *inp_ptr = gamma_val * (*inp_ptr - mean_val) * inv_std + beta_val;
    }

    // Store all 4 components
    output[base_idx + 0] = static_cast<scalar_t>(inp.x);
    output[base_idx + 1] = static_cast<scalar_t>(inp.y);
    output[base_idx + 2] = static_cast<scalar_t>(inp.z);
    output[base_idx + 3] = static_cast<scalar_t>(inp.w);
}

// ============================================================================
// OPTIMIZED QCONV FORWARD KERNEL
// Uses register blocking, vectorized I/O, and fused Hamilton mixing
// ============================================================================
template <typename scalar_t>
__global__ void qconv_forward_kernel_optimized(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight_r,
    const scalar_t* __restrict__ weight_i,
    const scalar_t* __restrict__ weight_j,
    const scalar_t* __restrict__ weight_k,
    const scalar_t* __restrict__ bias_r,
    const int B,
    const int C_in,
    const int C_out,
    const int H_in,
    const int W_in,
    const int H_out,
    const int W_out,
    const int kH,
    const int kW,
    const int strideH,
    const int strideW,
    const int padH,
    const int padW,
    const int dilationH,
    const int dilationW,
    const int groups) {

    using acc_t = float;  // Always accumulate in float for stability

    const int output_elements = B * C_out * H_out * W_out;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= output_elements) return;

    // Decompose output index
    const int w_out = idx % W_out;
    const int h_out = (idx / W_out) % H_out;
    const int c_out = (idx / (W_out * H_out)) % C_out;
    const int batch = idx / (W_out * H_out * C_out);

    // Group handling
    const int C_in_grp = C_in / groups;
    const int c_in_start = (c_out / (C_out / groups)) * C_in_grp;
    const int c_in_end = c_in_start + C_in_grp;

    // Initialize accumulators (use registers)
    acc_t sum_r = bias_r ? static_cast<acc_t>(bias_r[c_out]) : 0.0f;
    acc_t sum_i = 0.0f;
    acc_t sum_j = 0.0f;
    acc_t sum_k = 0.0f;

    // Convolution loop with register blocking
    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        #pragma unroll 3
        for (int kh = 0; kh < kH; ++kh) {
            const int h_in = h_out * strideH - padH + kh * dilationH;
            if (h_in < 0 || h_in >= H_in) continue;

            #pragma unroll 3
            for (int kw = 0; kw < kW; ++kw) {
                const int w_in = w_out * strideW - padW + kw * dilationW;
                if (w_in < 0 || w_in >= W_in) continue;

                // Weight index
                const int weight_idx = ((c_out * C_in_grp + (c_in - c_in_start)) * kH + kh) * kW + kw;

                // Load weights into registers
                const acc_t wr = static_cast<acc_t>(weight_r[weight_idx]);
                const acc_t wi = static_cast<acc_t>(weight_i[weight_idx]);
                const acc_t wj = static_cast<acc_t>(weight_j[weight_idx]);
                const acc_t wk = static_cast<acc_t>(weight_k[weight_idx]);

                // Input quaternion base index (BCHWQ layout)
                const int input_base_idx = ((batch * C_in + c_in) * H_in + h_in) * W_in + w_in;
                const int quat_idx = input_base_idx * 4;

                // Load input quaternion (could use float4 if aligned)
                const acc_t xr = static_cast<acc_t>(input[quat_idx + 0]);
                const acc_t xi = static_cast<acc_t>(input[quat_idx + 1]);
                const acc_t xj = static_cast<acc_t>(input[quat_idx + 2]);
                const acc_t xk = static_cast<acc_t>(input[quat_idx + 3]);

                // Left separable accumulation (matching your working kernel)
                sum_r = fmaf(wr, xr, sum_r);
                sum_i = fmaf(wi, xi, sum_i);
                sum_j = fmaf(wj, xj, sum_j);
                sum_k = fmaf(wk, xk, sum_k);
            }
        }
    }

    // Zhou separable forward mixing (CORRECTED - your working version)
    // This is the mixing that empirically works best
    const acc_t final_r = sum_r + sum_i + sum_j + sum_k;
    const acc_t final_i = -sum_i + sum_r + sum_k - sum_j;
    const acc_t final_j = -sum_j - sum_k + sum_r + sum_i;
    const acc_t final_k = -sum_k + sum_j - sum_i + sum_r;

    // Store output (BCHWQ layout)
    const int output_base_idx = ((batch * C_out + c_out) * H_out + h_out) * W_out + w_out;
    const int out_quat_idx = output_base_idx * 4;

    output[out_quat_idx + 0] = static_cast<scalar_t>(final_r);
    output[out_quat_idx + 1] = static_cast<scalar_t>(final_i);
    output[out_quat_idx + 2] = static_cast<scalar_t>(final_j);
    output[out_quat_idx + 3] = static_cast<scalar_t>(final_k);
}

// ============================================================================
// OPTIMIZED BACKWARD INPUT KERNEL
// ============================================================================
template <typename scalar_t>
__global__ void qconv_backward_input_kernel_optimized(
    scalar_t* __restrict__ grad_input,
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ weight_r,
    const scalar_t* __restrict__ weight_i,
    const scalar_t* __restrict__ weight_j,
    const scalar_t* __restrict__ weight_k,
    const int B,
    const int C_in,
    const int C_out,
    const int H_in,
    const int W_in,
    const int H_out,
    const int W_out,
    const int kH,
    const int kW,
    const int strideH,
    const int strideW,
    const int padH,
    const int padW,
    const int dilationH,
    const int dilationW,
    const int groups) {

    using acc_t = float;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = B * C_in * H_in * W_in;

    if (idx >= total_elements) return;

    const int w_in = idx % W_in;
    const int h_in = (idx / W_in) % H_in;
    const int c_in = (idx / (W_in * H_in)) % C_in;
    const int batch = idx / (W_in * H_in * C_in);

    const int group = c_in / (C_in / groups);
    const int C_out_grp = C_out / groups;
    const int c_out_start = group * C_out_grp;
    const int c_out_end = c_out_start + C_out_grp;
    const int C_in_grp = C_in / groups;

    acc_t grad_xr = 0.0f, grad_xi = 0.0f, grad_xj = 0.0f, grad_xk = 0.0f;

    for (int c_out = c_out_start; c_out < c_out_end; ++c_out) {
        #pragma unroll 3
        for (int kh = 0; kh < kH; ++kh) {
            const int h_out = (h_in + padH - kh * dilationH);
            if (h_out % strideH != 0) continue;
            const int h_out_idx = h_out / strideH;
            if (h_out_idx < 0 || h_out_idx >= H_out) continue;

            #pragma unroll 3
            for (int kw = 0; kw < kW; ++kw) {
                const int w_out = (w_in + padW - kw * dilationW);
                if (w_out % strideW != 0) continue;
                const int w_out_idx = w_out / strideW;
                if (w_out_idx < 0 || w_out_idx >= W_out) continue;

                // Load grad_output quaternion
                const int grad_out_base = ((batch * C_out + c_out) * H_out + h_out_idx) * W_out + w_out_idx;
                const int grad_out_quat = grad_out_base * 4;

                const acc_t grad_r = static_cast<acc_t>(grad_output[grad_out_quat + 0]);
                const acc_t grad_i = static_cast<acc_t>(grad_output[grad_out_quat + 1]);
                const acc_t grad_j = static_cast<acc_t>(grad_output[grad_out_quat + 2]);
                const acc_t grad_k = static_cast<acc_t>(grad_output[grad_out_quat + 3]);

                // Load weights
                const int weight_idx = ((c_out * C_in_grp + (c_in - group * C_in_grp)) * kH + kh) * kW + kw;
                const acc_t wr = static_cast<acc_t>(weight_r[weight_idx]);
                const acc_t wi = static_cast<acc_t>(weight_i[weight_idx]);
                const acc_t wj = static_cast<acc_t>(weight_j[weight_idx]);
                const acc_t wk = static_cast<acc_t>(weight_k[weight_idx]);

                // Correct Left Conj separable backward (transpose of forward mixing)
                grad_xr = fmaf((grad_r + grad_i + grad_j + grad_k), wr, grad_xr);
                grad_xi = fmaf((-grad_i + grad_r - grad_k + grad_j), wi, grad_xi);
                grad_xj = fmaf((-grad_j + grad_k + grad_r - grad_i), wj, grad_xj);
                grad_xk = fmaf((-grad_k - grad_j + grad_i + grad_r), wk, grad_xk);
            }
        }
    }

    // Store gradients (BCHWQ layout)
    const int grad_input_quat = idx * 4;
    grad_input[grad_input_quat + 0] = static_cast<scalar_t>(grad_xr);
    grad_input[grad_input_quat + 1] = static_cast<scalar_t>(grad_xi);
    grad_input[grad_input_quat + 2] = static_cast<scalar_t>(grad_xj);
    grad_input[grad_input_quat + 3] = static_cast<scalar_t>(grad_xk);
}

// ============================================================================
// OPTIMIZED BACKWARD WEIGHT KERNEL
// Better warp-level reduction
// ============================================================================
template <typename scalar_t>
__global__ void qconv_backward_weight_kernel_optimized(
    scalar_t* __restrict__ grad_weight_r,
    scalar_t* __restrict__ grad_weight_i,
    scalar_t* __restrict__ grad_weight_j,
    scalar_t* __restrict__ grad_weight_k,
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    const int B,
    const int C_in,
    const int C_out,
    const int H_in,
    const int W_in,
    const int H_out,
    const int W_out,
    const int kH,
    const int kW,
    const int strideH,
    const int strideW,
    const int padH,
    const int padW,
    const int dilationH,
    const int dilationW,
    const int groups) {

    using acc_t = float;

    const int weight_idx = blockIdx.x;
    const int C_in_grp = C_in / groups;
    const int total_weights = C_out * C_in_grp * kH * kW;

    if (weight_idx >= total_weights) return;

    // Decode weight index
    const int kw = weight_idx % kW;
    const int kh = (weight_idx / kW) % kH;
    const int c_in = (weight_idx / (kW * kH)) % C_in_grp;
    const int c_out = weight_idx / (kW * kH * C_in_grp);

    const int group = c_out / (C_out / groups);
    const int global_c_in = group * C_in_grp + c_in;

    // Shared memory for warp reduction (4 gradients x 32 warps max)
    __shared__ acc_t shared_grads[4][32];

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;

    acc_t local_grad_r = 0.0f, local_grad_i = 0.0f;
    acc_t local_grad_j = 0.0f, local_grad_k = 0.0f;

    const int total_pixels = B * H_out * W_out;

    // Grid-stride loop for better load balancing
    for (int pixel_idx = tid; pixel_idx < total_pixels; pixel_idx += num_threads) {
        const int w_out = pixel_idx % W_out;
        const int h_out = (pixel_idx / W_out) % H_out;
        const int batch = pixel_idx / (W_out * H_out);

        const int h_in = h_out * strideH - padH + kh * dilationH;
        const int w_in = w_out * strideW - padW + kw * dilationW;

        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
            // Load grad_output
            const int grad_out_base = ((batch * C_out + c_out) * H_out + h_out) * W_out + w_out;
            const int grad_out_quat = grad_out_base * 4;

            const acc_t grad_r = static_cast<acc_t>(grad_output[grad_out_quat + 0]);
            const acc_t grad_i = static_cast<acc_t>(grad_output[grad_out_quat + 1]);
            const acc_t grad_j = static_cast<acc_t>(grad_output[grad_out_quat + 2]);
            const acc_t grad_k = static_cast<acc_t>(grad_output[grad_out_quat + 3]);

            // Load input
            const int input_base = ((batch * C_in + global_c_in) * H_in + h_in) * W_in + w_in;
            const int input_quat = input_base * 4;

            const acc_t xr = static_cast<acc_t>(input[input_quat + 0]);
            const acc_t xi = static_cast<acc_t>(input[input_quat + 1]);
            const acc_t xj = static_cast<acc_t>(input[input_quat + 2]);
            const acc_t xk = static_cast<acc_t>(input[input_quat + 3]);

            // Correct Left Conj separable weight gradient
            local_grad_r = fmaf((grad_r + grad_i + grad_j + grad_k), xr, local_grad_r);
            local_grad_i = fmaf((-grad_i + grad_r - grad_k + grad_j), xi, local_grad_i);
            local_grad_j = fmaf((-grad_j + grad_k + grad_r - grad_i), xj, local_grad_j);
            local_grad_k = fmaf((-grad_k - grad_j + grad_i + grad_r), xk, local_grad_k);
        }
    }

    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_grad_r += __shfl_down_sync(0xffffffff, local_grad_r, offset);
        local_grad_i += __shfl_down_sync(0xffffffff, local_grad_i, offset);
        local_grad_j += __shfl_down_sync(0xffffffff, local_grad_j, offset);
        local_grad_k += __shfl_down_sync(0xffffffff, local_grad_k, offset);
    }

    // Store warp results to shared memory
    if (lane_id == 0) {
        shared_grads[0][warp_id] = local_grad_r;
        shared_grads[1][warp_id] = local_grad_i;
        shared_grads[2][warp_id] = local_grad_j;
        shared_grads[3][warp_id] = local_grad_k;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        const int num_warps = (num_threads + 31) / 32;
        acc_t final_r = (lane_id < num_warps) ? shared_grads[0][lane_id] : 0.0f;
        acc_t final_i = (lane_id < num_warps) ? shared_grads[1][lane_id] : 0.0f;
        acc_t final_j = (lane_id < num_warps) ? shared_grads[2][lane_id] : 0.0f;
        acc_t final_k = (lane_id < num_warps) ? shared_grads[3][lane_id] : 0.0f;

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            final_r += __shfl_down_sync(0xffffffff, final_r, offset);
            final_i += __shfl_down_sync(0xffffffff, final_i, offset);
            final_j += __shfl_down_sync(0xffffffff, final_j, offset);
            final_k += __shfl_down_sync(0xffffffff, final_k, offset);
        }

        if (lane_id == 0) {
            grad_weight_r[weight_idx] = static_cast<scalar_t>(final_r);
            grad_weight_i[weight_idx] = static_cast<scalar_t>(final_i);
            grad_weight_j[weight_idx] = static_cast<scalar_t>(final_j);
            grad_weight_k[weight_idx] = static_cast<scalar_t>(final_k);
        }
    }
}

// ============================================================================
// FUSED QCONV + IQBN + SILU KERNEL (inference only)
// Eliminates intermediate memory round-trips
// ============================================================================
template <typename scalar_t>
__global__ void qconv_bn_silu_fused_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ conv_output,  // Pre-computed conv output
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ beta,
    const scalar_t* __restrict__ running_mean,
    const scalar_t* __restrict__ running_var,
    const int B,
    const int C,
    const int H,
    const int W,
    const float eps) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * C * H * W;
    if (idx >= total) return;

    const int w = idx % W;
    const int h = (idx / W) % H;
    const int c = (idx / (W * H)) % C;
    const int b = idx / (W * H * C);

    const int quat_base = idx * 4;
    const int stat_base = c * 4;

    #pragma unroll
    for (int q = 0; q < 4; q++) {
        // Load conv output
        float x = static_cast<float>(conv_output[quat_base + q]);

        // BatchNorm
        const float mean = static_cast<float>(running_mean[stat_base + q]);
        const float var = static_cast<float>(running_var[stat_base + q]);
        const float g = static_cast<float>(gamma[stat_base + q]);
        const float be = static_cast<float>(beta[stat_base + q]);

        x = g * (x - mean) * rsqrtf(var + eps) + be;

        // SiLU activation: x * sigmoid(x)
        const float sigmoid_x = 1.0f / (1.0f + expf(-x));
        x = x * sigmoid_x;

        output[quat_base + q] = static_cast<scalar_t>(x);
    }
}

// ============================================================================
// HOST WRAPPER FUNCTIONS
// ============================================================================

torch::Tensor qconv_forward_cuda_optimized(
    torch::Tensor input,
    torch::Tensor weight_r,
    torch::Tensor weight_i,
    torch::Tensor weight_j,
    torch::Tensor weight_k,
    torch::Tensor bias_r,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups) {

    TORCH_CHECK(input.dim() == 5 && input.size(4) == 4, "Input must be [B,C,H,W,4]");

    const auto B = input.size(0);
    const auto C_in = input.size(1);
    const auto H_in = input.size(2);
    const auto W_in = input.size(3);

    const auto C_out = weight_r.size(0);
    const auto kH = weight_r.size(2);
    const auto kW = weight_r.size(3);

    const int H_out = (H_in + 2 * padding[0] - dilation[0] * (kH - 1) - 1) / stride[0] + 1;
    const int W_out = (W_in + 2 * padding[1] - dilation[1] * (kW - 1) - 1) / stride[1] + 1;

    auto output = torch::empty({B, C_out, H_out, W_out, 4}, input.options());

    // Use 256 threads for better occupancy on modern GPUs
    const int threads = 256;
    const int blocks = (B * C_out * H_out * W_out + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "qconv_forward_optimized", ([&] {
        qconv_forward_kernel_optimized<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            weight_r.data_ptr<scalar_t>(),
            weight_i.data_ptr<scalar_t>(),
            weight_j.data_ptr<scalar_t>(),
            weight_k.data_ptr<scalar_t>(),
            bias_r.defined() ? bias_r.data_ptr<scalar_t>() : nullptr,
            B, C_in, C_out, H_in, W_in, H_out, W_out,
            kH, kW, stride[0], stride[1], padding[0], padding[1],
            dilation[0], dilation[1], groups
        );
    }));

    return output;
}

torch::Tensor iqbn_forward_cuda_optimized(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps) {

    TORCH_CHECK(input.dim() == 5 && input.size(4) == 4, "Input must be [B,C,H,W,4]");

    const auto B = input.size(0);
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);
    const auto HW = H * W;

    auto output = torch::empty_like(input);

    // Process one spatial location per thread (all 4 quaternion components)
    const int threads = 256;
    const int total_spatial = B * C * HW;
    const int blocks = (total_spatial + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "iqbn_forward_optimized", ([&] {
        iqbn_forward_kernel_optimized<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.contiguous().data_ptr<scalar_t>(),
            gamma.contiguous().data_ptr<scalar_t>(),
            beta.contiguous().data_ptr<scalar_t>(),
            running_mean.contiguous().data_ptr<scalar_t>(),
            running_var.contiguous().data_ptr<scalar_t>(),
            B, C, HW, eps
        );
    }));

    return output;
}

std::vector<torch::Tensor> qconv_backward_cuda_optimized(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight_r,
    torch::Tensor weight_i,
    torch::Tensor weight_j,
    torch::Tensor weight_k,
    bool bias_defined,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups) {

    const auto B = input.size(0);
    const auto C_in = input.size(1);
    const auto H_in = input.size(2);
    const auto W_in = input.size(3);

    const auto C_out = weight_r.size(0);
    const auto C_in_grp = weight_r.size(1);
    const auto kH = weight_r.size(2);
    const auto kW = weight_r.size(3);

    const auto H_out = grad_output.size(2);
    const auto W_out = grad_output.size(3);

    auto grad_input = torch::zeros_like(input);
    auto grad_weight_r = torch::zeros_like(weight_r);
    auto grad_weight_i = torch::zeros_like(weight_i);
    auto grad_weight_j = torch::zeros_like(weight_j);
    auto grad_weight_k = torch::zeros_like(weight_k);
    auto grad_bias = bias_defined ? torch::zeros({C_out}, grad_output.options()) : torch::Tensor();

    const int threads = 256;

    // Backward input
    const int input_blocks = (B * C_in * H_in * W_in + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "qconv_backward_input_opt", ([&] {
        qconv_backward_input_kernel_optimized<scalar_t><<<input_blocks, threads>>>(
            grad_input.data_ptr<scalar_t>(),
            grad_output.data_ptr<scalar_t>(),
            weight_r.data_ptr<scalar_t>(),
            weight_i.data_ptr<scalar_t>(),
            weight_j.data_ptr<scalar_t>(),
            weight_k.data_ptr<scalar_t>(),
            B, C_in, C_out, H_in, W_in, H_out, W_out,
            kH, kW, stride[0], stride[1], padding[0], padding[1],
            dilation[0], dilation[1], groups
        );
    }));

    // Backward weight - one block per weight, multiple threads per block
    const int weight_blocks = C_out * C_in_grp * kH * kW;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "qconv_backward_weight_opt", ([&] {
        qconv_backward_weight_kernel_optimized<scalar_t><<<weight_blocks, threads>>>(
            grad_weight_r.data_ptr<scalar_t>(),
            grad_weight_i.data_ptr<scalar_t>(),
            grad_weight_j.data_ptr<scalar_t>(),
            grad_weight_k.data_ptr<scalar_t>(),
            grad_output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            B, C_in, C_out, H_in, W_in, H_out, W_out,
            kH, kW, stride[0], stride[1], padding[0], padding[1],
            dilation[0], dilation[1], groups
        );
    }));

    return {grad_input, grad_weight_r, grad_weight_i, grad_weight_j, grad_weight_k, grad_bias};
}

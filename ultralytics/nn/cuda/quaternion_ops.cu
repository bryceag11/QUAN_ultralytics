// quaternion_ops.cu - Implement baked-in Hamilton product
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void iqbn_forward_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ beta,
    const scalar_t* __restrict__ running_mean,
    const scalar_t* __restrict__ running_var,
    const int N,        
    const int C_per_q,  
    const int Q,          
    const int HW,         
    const float eps) {

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N * C_per_q * Q * HW) return;

    // Calculate indices
    const int hw_idx = index % HW;
    const int q = (index / HW) % Q;
    const int c = (index / (HW * Q)) % C_per_q;

    const int stat_idx = c * Q + q; 

    const scalar_t mean = running_mean[stat_idx];
    const scalar_t var = running_var[stat_idx];
    const scalar_t gamma_val = gamma[stat_idx];
    const scalar_t beta_val = beta[stat_idx];

    const scalar_t inp = input[index];
    output[index] = gamma_val * (inp - mean) / sqrt(var + eps) + beta_val;
}


template <typename scalar_t>
__global__ void qconv_forward_kernel_hamilton(
    scalar_t* __restrict__ output,       // Shape: [B, C_out_per_q, Q=4, H_out, W_out]
    const scalar_t* __restrict__ input,  // Shape: [B, C_in_per_q, Q=4, H_in, W_in]
    // Separate weights for R, I, J, K components of the filter
    const scalar_t* __restrict__ weight_r, // Shape: [C_out_per_q, C_in_per_q_grp, kH, kW]
    const scalar_t* __restrict__ weight_i, // Shape: [C_out_per_q, C_in_per_q_grp, kH, kW]
    const scalar_t* __restrict__ weight_j, // Shape: [C_out_per_q, C_in_per_q_grp, kH, kW]
    const scalar_t* __restrict__ weight_k, // Shape: [C_out_per_q, C_in_per_q_grp, kH, kW]
    // Separate biases for R, I, J, K components of the output
    const scalar_t* __restrict__ bias_r,   // Shape: [C_out_per_q] (or null)
    const scalar_t* __restrict__ bias_i,   // Shape: [C_out_per_q] (or null)
    const scalar_t* __restrict__ bias_j,   // Shape: [C_out_per_q] (or null)
    const scalar_t* __restrict__ bias_k,   // Shape: [C_out_per_q] (or null)
    const int B,
    const int C_in_per_q,  // Input channels per quaternion component
    const int C_out_per_q, // Output channels per quaternion component
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
    const int groups
    ) {

    // Output element index calculation (1D grid-stride loop)
    // Total output elements = B * C_out_per_q * H_out * W_out (compute one output quaternion (r,i,j,k) per thread)
    const int Q = 4; // Quaternion dimension is fixed
    const int output_quaternion_elements = B * C_out_per_q * H_out * W_out;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= output_quaternion_elements) return;

    // Deconstruct index to find batch, output channel, output H/W
    const int w_out = idx % W_out;
    const int h_out = (idx / W_out) % H_out;
    const int c_out = (idx / (W_out * H_out)) % C_out_per_q; // Index for the output quaternion channel group
    const int batch = idx / (W_out * H_out * C_out_per_q);

    // Calculate input channel range
    const int C_in_per_q_grp = C_in_per_q / groups; // Input channels per component per group
    const int c_in_start = (c_out / (C_out_per_q / groups)) * C_in_per_q_grp;
    const int c_in_end = c_in_start + C_in_per_q_grp;

    // Accumulators for the output quaternion components
    scalar_t sum_r = bias_r ? bias_r[c_out] : 0;
    scalar_t sum_i = bias_i ? bias_i[c_out] : 0;
    scalar_t sum_j = bias_j ? bias_j[c_out] : 0;
    scalar_t sum_k = bias_k ? bias_k[c_out] : 0;

    // --- Convolution loops ---
    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) { // Loop over input channels per component within the group
        // Adjust c_in relative to the group start for weight indexing
        const int c_in_grp_offset = c_in - c_in_start;

        for (int kh = 0; kh < kH; ++kh) {
            const int h_in = h_out * strideH - padH + kh * dilationH;

            // Check height boundary
            if (h_in < 0 || h_in >= H_in) continue;

            for (int kw = 0; kw < kW; ++kw) {
                const int w_in = w_out * strideW - padW + kw * dilationW;

                // Check width boundary
                if (w_in < 0 || w_in >= W_in) continue;

                // --- Calculate indices for input and weights ---
                // Input index base for this spatial location and input channel component
                const int input_base_idx = ((batch * C_in_per_q + c_in) * H_in + h_in) * W_in + w_in;
                // Weight index base for this output channel, input channel (group relative), and kernel location
                const int weight_base_idx = ((c_out * C_in_per_q_grp + c_in_grp_offset) * kH + kh) * kW + kw;

                // --- Fetch input quaternion components (x_r, x_i, x_j, x_k) ---
                // Input layout: [B, C_in_per_q, Q=4, H_in, W_in] -> flat access needs strides
                const int stride_B_in = C_in_per_q * Q * H_in * W_in;
                const int stride_C_in = Q * H_in * W_in;
                const int stride_Q_in = H_in * W_in;
                const int stride_H_in = W_in;
                const int stride_W_in = 1;

                const int base_offset_in = batch * stride_B_in + c_in * stride_C_in + h_in * stride_H_in + w_in * stride_W_in;
                const scalar_t xr = input[base_offset_in + 0 * stride_Q_in]; // q=0 is R
                const scalar_t xi = input[base_offset_in + 1 * stride_Q_in]; // q=1 is I
                const scalar_t xj = input[base_offset_in + 2 * stride_Q_in]; // q=2 is J
                const scalar_t xk = input[base_offset_in + 3 * stride_Q_in]; // q=3 is K

                // --- Fetch weight quaternion components (w_r, w_i, w_j, w_k) ---
                // Weight layout assumed: [C_out_per_q, C_in_per_q_grp, kH, kW] for each of the 4 weight tensors
                const int stride_C_out_w = C_in_per_q_grp * kH * kW;
                const int stride_C_in_w = kH * kW;
                const int stride_kH_w = kW;
                const int stride_kW_w = 1;
                const int base_offset_w = c_out * stride_C_out_w + c_in_grp_offset * stride_C_in_w + kh * stride_kH_w + kw * stride_kW_w;

                const scalar_t wr = weight_r[base_offset_w];
                const scalar_t wi = weight_i[base_offset_w];
                const scalar_t wj = weight_j[base_offset_w];
                const scalar_t wk = weight_k[base_offset_w];

                // --- Hamilton Product and Accumulation ---
                sum_r += xr * wr - xi * wi - xj * wj - xk * wk;
                sum_i += xr * wr + xi * wi + xj * wj - xk * wk;
                sum_j += xr * wr - xi * wi + xj * wj + xk * wk;
                sum_k += xr * wr + xi * wi - xj * wj + xk * wk;
            } // kw loop
        } // kh loop
    } // c_in loop

    // --- Write output quaternion components ---
    // Output layout: [B, C_out_per_q, Q=4, H_out, W_out]
    const int stride_B_out = C_out_per_q * Q * H_out * W_out;
    const int stride_C_out = Q * H_out * W_out;
    const int stride_Q_out = H_out * W_out;
    const int stride_H_out = W_out;
    const int stride_W_out = 1;

    const int base_offset_out = batch * stride_B_out + c_out * stride_C_out + h_out * stride_H_out + w_out * stride_W_out;

    output[base_offset_out + 0 * stride_Q_out] = sum_r; // q=0 is R
    output[base_offset_out + 1 * stride_Q_out] = sum_i; // q=1 is I
    output[base_offset_out + 2 * stride_Q_out] = sum_j; // q=2 is J
    output[base_offset_out + 3 * stride_Q_out] = sum_k; // q=3 is K
}


// --- CUDA function implementations ---
torch::Tensor iqbn_forward_cuda(
    torch::Tensor input,        // [B, C_per_q, Q, H, W]
    torch::Tensor gamma,        // [C_per_q, Q]
    torch::Tensor beta,         // [C_per_q, Q]
    torch::Tensor running_mean, // [C_per_q, Q]
    torch::Tensor running_var,  // [C_per_q, Q]
    float eps) {

    TORCH_CHECK(input.dim() == 5, "iqbn_forward_cuda expects 5D input");
    TORCH_CHECK(gamma.dim() == 2, "iqbn_forward_cuda expects 2D gamma");
    TORCH_CHECK(beta.dim() == 2, "iqbn_forward_cuda expects 2D beta");
    TORCH_CHECK(running_mean.dim() == 2, "iqbn_forward_cuda expects 2D running_mean");
    TORCH_CHECK(running_var.dim() == 2, "iqbn_forward_cuda expects 2D running_var");

    const auto N = input.size(0);
    const auto C_per_q = input.size(1);
    const auto Q = input.size(2);
    const auto H = input.size(3);
    const auto W = input.size(4);
    const auto HW = H * W;

    TORCH_CHECK(Q == 4, "Input quaternion dimension (dim 2) must be 4");
    TORCH_CHECK(gamma.size(0) == C_per_q && gamma.size(1) == Q, "gamma shape mismatch");
    TORCH_CHECK(beta.size(0) == C_per_q && beta.size(1) == Q, "beta shape mismatch");
    TORCH_CHECK(running_mean.size(0) == C_per_q && running_mean.size(1) == Q, "running_mean shape mismatch");
    TORCH_CHECK(running_var.size(0) == C_per_q && running_var.size(1) == Q, "running_var shape mismatch");


    auto output = torch::empty_like(input);

    const int threads = 256;
    const int total_elements = N * C_per_q * Q * HW;
    const int blocks = (total_elements + threads - 1) / threads;

    // Ensure tensors are contiguous for direct data pointer access in kernel
    input = input.contiguous();
    gamma = gamma.contiguous();
    beta = beta.contiguous();
    running_mean = running_mean.contiguous();
    running_var = running_var.contiguous();


    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "iqbn_forward_cuda", ([&] {
        iqbn_forward_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(),
            beta.data_ptr<scalar_t>(),
            running_mean.data_ptr<scalar_t>(),
            running_var.data_ptr<scalar_t>(),
            N, C_per_q, Q, HW, eps
        );
    }));

    return output;
}

// --- Updated qconv_forward_cuda to accept 4 weights/biases ---
torch::Tensor qconv_forward_cuda(
    torch::Tensor input,        // [B, C_in_per_q, Q=4, H_in, W_in]
    torch::Tensor weight_r,     // [C_out_per_q, C_in_per_q_grp, kH, kW]
    torch::Tensor weight_i,     // [C_out_per_q, C_in_per_q_grp, kH, kW]
    torch::Tensor weight_j,     // [C_out_per_q, C_in_per_q_grp, kH, kW]
    torch::Tensor weight_k,     // [C_out_per_q, C_in_per_q_grp, kH, kW]
    torch::Tensor bias_r,       // [C_out_per_q] or None
    torch::Tensor bias_i,       // [C_out_per_q] or None
    torch::Tensor bias_j,       // [C_out_per_q] or None
    torch::Tensor bias_k,       // [C_out_per_q] or None
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups) {

    TORCH_CHECK(input.dim() == 5, "qconv_forward_cuda expects 5D input");
    TORCH_CHECK(input.size(2) == 4, "Input quaternion dimension (dim 2) must be 4");
    TORCH_CHECK(weight_r.dim() == 4, "weight_r must be 4D");
    TORCH_CHECK(weight_i.dim() == 4, "weight_i must be 4D");
    TORCH_CHECK(weight_j.dim() == 4, "weight_j must be 4D");
    TORCH_CHECK(weight_k.dim() == 4, "weight_k must be 4D");

    const auto B = input.size(0);
    const auto C_in_per_q = input.size(1);
    const auto H_in = input.size(3);
    const auto W_in = input.size(4);

    const auto C_out_per_q = weight_r.size(0); // Output channels per component
    const auto C_in_per_q_grp = weight_r.size(1); // Input channels per component per group
    const auto kH = weight_r.size(2);
    const auto kW = weight_r.size(3);

    // Basic shape consistency checks
    TORCH_CHECK(C_in_per_q == C_in_per_q_grp * groups, "Input channels, group channels, and groups mismatch");
    TORCH_CHECK(weight_i.sizes() == weight_r.sizes(), "weight_i shape mismatch");
    TORCH_CHECK(weight_j.sizes() == weight_r.sizes(), "weight_j shape mismatch");
    TORCH_CHECK(weight_k.sizes() == weight_r.sizes(), "weight_k shape mismatch");

    if (bias_r.defined()) {
        TORCH_CHECK(bias_r.dim() == 1 && bias_r.size(0) == C_out_per_q, "bias_r shape mismatch");
        TORCH_CHECK(bias_i.defined() && bias_i.sizes() == bias_r.sizes(), "bias_i shape mismatch");
        TORCH_CHECK(bias_j.defined() && bias_j.sizes() == bias_r.sizes(), "bias_j shape mismatch");
        TORCH_CHECK(bias_k.defined() && bias_k.sizes() == bias_r.sizes(), "bias_k shape mismatch");
    }


    const auto H_out = (H_in + 2 * padding[0] - dilation[0] * (kH - 1) - 1) / stride[0] + 1;
    const auto W_out = (W_in + 2 * padding[1] - dilation[1] * (kW - 1) - 1) / stride[1] + 1;

    // Output shape: [B, C_out_per_q, Q=4, H_out, W_out]
    auto output = torch::empty({B, C_out_per_q, 4, H_out, W_out}, input.options());

    // Optimized launch parameters using 1D thread blocks
    const int threads = 256; // Adjust based on GPU architecture if needed
    const int total_output_quaternions = B * C_out_per_q * H_out * W_out;
    const int blocks = (total_output_quaternions + threads - 1) / threads;

    // Ensure contiguous inputs for direct data pointer access
    input = input.contiguous();
    weight_r = weight_r.contiguous();
    weight_i = weight_i.contiguous();
    weight_j = weight_j.contiguous();
    weight_k = weight_k.contiguous();
    bias_r = bias_r.defined() ? bias_r.contiguous() : bias_r;
    bias_i = bias_i.defined() ? bias_i.contiguous() : bias_i;
    bias_j = bias_j.defined() ? bias_j.contiguous() : bias_j;
    bias_k = bias_k.defined() ? bias_k.contiguous() : bias_k;


    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "qconv_forward_cuda_hamilton", ([&] {
        qconv_forward_kernel_hamilton<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            weight_r.data_ptr<scalar_t>(),
            weight_i.data_ptr<scalar_t>(),
            weight_j.data_ptr<scalar_t>(),
            weight_k.data_ptr<scalar_t>(),
            bias_r.defined() ? bias_r.data_ptr<scalar_t>() : nullptr,
            bias_i.defined() ? bias_i.data_ptr<scalar_t>() : nullptr,
            bias_j.defined() ? bias_j.data_ptr<scalar_t>() : nullptr,
            bias_k.defined() ? bias_k.data_ptr<scalar_t>() : nullptr,
            B, C_in_per_q, C_out_per_q, H_in, W_in, H_out, W_out,
            kH, kW, stride[0], stride[1], padding[0], padding[1],
            dilation[0], dilation[1], // Pass dilation
            groups
        );
    }));

    return output;
}

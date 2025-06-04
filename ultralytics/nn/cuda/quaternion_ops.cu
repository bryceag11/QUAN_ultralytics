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
// Fixed forward kernel - removed unused stride variables

template <typename scalar_t>
__global__ void qconv_forward_kernel_hamilton(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight_r,
    const scalar_t* __restrict__ weight_i,
    const scalar_t* __restrict__ weight_j,
    const scalar_t* __restrict__ weight_k,
    const scalar_t* __restrict__ bias_r,
    const scalar_t* __restrict__ bias_i,
    const scalar_t* __restrict__ bias_j,
    const scalar_t* __restrict__ bias_k,
    const int B,
    const int C_in_per_q,
    const int C_out_per_q,
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
    
    using acc_scalar_t = typename std::conditional<std::is_same<scalar_t, double>::value, double, float>::type;

    const int Q = 4;
    const int output_quaternion_elements = B * C_out_per_q * H_out * W_out;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= output_quaternion_elements) return;

    const int w_out = idx % W_out;
    const int h_out = (idx / W_out) % H_out;
    const int c_out = (idx / (W_out * H_out)) % C_out_per_q;
    const int batch = idx / (W_out * H_out * C_out_per_q);

    const int C_in_per_q_grp = C_in_per_q / groups;
    const int c_in_start = (c_out / (C_out_per_q / groups)) * C_in_per_q_grp;
    const int c_in_end = c_in_start + C_in_per_q_grp;

    // Use acc_scalar_t for accumulators
    acc_scalar_t sum_r_acc = bias_r ? static_cast<acc_scalar_t>(bias_r[c_out]) : 0.0;
    acc_scalar_t sum_i_acc = 0.0;
    acc_scalar_t sum_j_acc = 0.0;
    acc_scalar_t sum_k_acc = 0.0;

    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        const int c_in_grp_offset = c_in - c_in_start;
        for (int kh = 0; kh < kH; ++kh) {
            const int h_in = h_out * strideH - padH + kh * dilationH;
            if (h_in < 0 || h_in >= H_in) continue;
            for (int kw = 0; kw < kW; ++kw) {
                const int w_in = w_out * strideW - padW + kw * dilationW;
                if (w_in < 0 || w_in >= W_in) continue;

                const int base_offset_in = batch * (C_in_per_q * Q * H_in * W_in) +
                                           c_in * (Q * H_in * W_in) +
                                           h_in * W_in +
                                           w_in;

                const scalar_t xr = input[base_offset_in + 0 * (H_in * W_in)];
                const scalar_t xi = input[base_offset_in + 1 * (H_in * W_in)];
                const scalar_t xj = input[base_offset_in + 2 * (H_in * W_in)];
                const scalar_t xk = input[base_offset_in + 3 * (H_in * W_in)];

                const int base_offset_w = c_out * (C_in_per_q_grp * kH * kW) +
                                          c_in_grp_offset * (kH * kW) +
                                          kh * kW +
                                          kw;

                const scalar_t wr = weight_r[base_offset_w];
                const scalar_t wi = weight_i[base_offset_w];
                const scalar_t wj = weight_j[base_offset_w];
                const scalar_t wk = weight_k[base_offset_w];

                sum_r_acc += static_cast<acc_scalar_t>(xr) * static_cast<acc_scalar_t>(wr);
                sum_i_acc += static_cast<acc_scalar_t>(xi) * static_cast<acc_scalar_t>(wi);
                sum_j_acc += static_cast<acc_scalar_t>(xj) * static_cast<acc_scalar_t>(wj);
                sum_k_acc += static_cast<acc_scalar_t>(xk) * static_cast<acc_scalar_t>(wk);
            }
        }
    }

    // Final combination
    acc_scalar_t final_r_acc = sum_r_acc + sum_i_acc + sum_j_acc + sum_k_acc;
    acc_scalar_t final_i_acc = -sum_r_acc + sum_i_acc + sum_j_acc - sum_k_acc;
    acc_scalar_t final_j_acc = -sum_r_acc - sum_i_acc + sum_j_acc + sum_k_acc;
    acc_scalar_t final_k_acc = -sum_r_acc + sum_i_acc - sum_j_acc + sum_k_acc;

    const int base_offset_out = batch * (C_out_per_q * Q * H_out * W_out) +
                                c_out * (Q * H_out * W_out) +
                                h_out * W_out +
                                w_out;

    output[base_offset_out + 0 * (H_out * W_out)] = static_cast<scalar_t>(final_r_acc);
    output[base_offset_out + 1 * (H_out * W_out)] = static_cast<scalar_t>(final_i_acc);
    output[base_offset_out + 2 * (H_out * W_out)] = static_cast<scalar_t>(final_j_acc);
    output[base_offset_out + 3 * (H_out * W_out)] = static_cast<scalar_t>(final_k_acc);
}

// Optimized backward input kernel - no shared memory, better coalescing
template <typename scalar_t>
__global__ void qconv_backward_input_kernel(
    scalar_t* __restrict__ grad_input,
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ weight_r,
    const scalar_t* __restrict__ weight_i,
    const scalar_t* __restrict__ weight_j,
    const scalar_t* __restrict__ weight_k,
    const int B,
    const int C_in_per_q,
    const int C_out_per_q,
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

    using acc_scalar_t = typename std::conditional<std::is_same<scalar_t, double>::value, double, float>::type;
    
    const int Q = 4;
    const int total_input_elements = B * C_in_per_q * H_in * W_in;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_input_elements) return;
    
    // Decode position
    const int w_in = idx % W_in;
    const int h_in = (idx / W_in) % H_in;
    const int c_in = (idx / (W_in * H_in)) % C_in_per_q;
    const int batch = idx / (W_in * H_in * C_in_per_q);
    
    // Group calculations
    const int C_out_per_q_grp = C_out_per_q / groups;
    const int C_in_per_q_grp = C_in_per_q / groups;
    const int group = c_in / C_in_per_q_grp;
    const int c_out_start = group * C_out_per_q_grp;
    const int c_out_end = c_out_start + C_out_per_q_grp;
    const int c_in_grp = c_in % C_in_per_q_grp;
    
    // Accumulators with higher precision
    acc_scalar_t grad_xr = 0, grad_xi = 0, grad_xj = 0, grad_xk = 0;
    
    // Process each output channel
    for (int c_out = c_out_start; c_out < c_out_end; ++c_out) {
        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                // Calculate corresponding output position
                const int h_out_base = h_in + padH - kh * dilationH;
                const int w_out_base = w_in + padW - kw * dilationW;
                
                if (h_out_base % strideH != 0 || w_out_base % strideW != 0) continue;
                
                const int h_out = h_out_base / strideH;
                const int w_out = w_out_base / strideW;
                
                if (h_out < 0 || h_out >= H_out || w_out < 0 || w_out >= W_out) continue;
                
                // Load gradients - coalesced access
                const int grad_base = ((batch * C_out_per_q + c_out) * Q) * H_out * W_out;
                const int grad_offset = h_out * W_out + w_out;
                
                const scalar_t grad_r = grad_output[grad_base + 0 * H_out * W_out + grad_offset];
                const scalar_t grad_i = grad_output[grad_base + 1 * H_out * W_out + grad_offset];
                const scalar_t grad_j = grad_output[grad_base + 2 * H_out * W_out + grad_offset];
                const scalar_t grad_k = grad_output[grad_base + 3 * H_out * W_out + grad_offset];
                
                // Load weights - coalesced access
                const int weight_idx = ((c_out * C_in_per_q_grp + c_in_grp) * kH + kh) * kW + kw;
                const scalar_t wr = weight_r[weight_idx];
                const scalar_t wi = weight_i[weight_idx];
                const scalar_t wj = weight_j[weight_idx];
                const scalar_t wk = weight_k[weight_idx];
                
                // Hamilton product backward (conjugate transpose)
                // grad_xr += static_cast<acc_scalar_t>(grad_r * wr - grad_i * wr - grad_j * wr - grad_k * wr);
                // grad_xi += static_cast<acc_scalar_t>(grad_r * wi + grad_i * wi - grad_j * wi + grad_k * wi);
                // grad_xj += static_cast<acc_scalar_t>(grad_r * wj + grad_i * wj + grad_j * wj - grad_k * wj);
                // grad_xk += static_cast<acc_scalar_t>(grad_r * wk - grad_i * wk + grad_j * wk + grad_k * wk);

                const acc_scalar_t combined_grad_for_r = grad_r - grad_i - grad_j - grad_k;
                const acc_scalar_t combined_grad_for_i = grad_r + grad_i - grad_j + grad_k;
                const acc_scalar_t combined_grad_for_j = grad_r + grad_i + grad_j - grad_k;
                const acc_scalar_t combined_grad_for_k = grad_r - grad_i + grad_j + grad_k;
                
                grad_xr += combined_grad_for_r * wr;
                grad_xi += combined_grad_for_i * wi;
                grad_xj += combined_grad_for_j * wj;
                grad_xk += combined_grad_for_k * wk;
            }
        }
    }
    
    // Write results - coalesced access
    const int input_base = ((batch * C_in_per_q + c_in) * Q) * H_in * W_in;
    const int input_offset = h_in * W_in + w_in;
    
    grad_input[input_base + 0 * H_in * W_in + input_offset] = static_cast<scalar_t>(grad_xr);
    grad_input[input_base + 1 * H_in * W_in + input_offset] = static_cast<scalar_t>(grad_xi);
    grad_input[input_base + 2 * H_in * W_in + input_offset] = static_cast<scalar_t>(grad_xj);
    grad_input[input_base + 3 * H_in * W_in + input_offset] = static_cast<scalar_t>(grad_xk);
}

// Highly optimized weight gradient kernel using warp-level primitives
template <typename scalar_t>
__global__ void qconv_backward_weight_kernel(
    scalar_t* __restrict__ grad_weight_r,
    scalar_t* __restrict__ grad_weight_i,
    scalar_t* __restrict__ grad_weight_j,
    scalar_t* __restrict__ grad_weight_k,
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    const int B,
    const int C_in_per_q,
    const int C_out_per_q,
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

    using acc_scalar_t = typename std::conditional<std::is_same<scalar_t, double>::value, double, float>::type;
    
    const int Q = 4;
    const int C_in_per_q_grp = C_in_per_q / groups;
    
    // One block per weight element
    const int total_weights = C_out_per_q * C_in_per_q_grp * kH * kW;
    const int weight_idx = blockIdx.x;
    
    if (weight_idx >= total_weights) return;
    
    // Decode weight position
    const int kw = weight_idx % kW;
    const int kh = (weight_idx / kW) % kH;
    const int c_in_grp = (weight_idx / (kW * kH)) % C_in_per_q_grp;
    const int c_out = weight_idx / (kW * kH * C_in_per_q_grp);
    
    const int group = c_out / (C_out_per_q / groups);
    const int c_in = group * C_in_per_q_grp + c_in_grp;
    
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    // Each thread processes multiple batch/spatial positions
    acc_scalar_t local_grad_r = 0, local_grad_i = 0, local_grad_j = 0, local_grad_k = 0;
    
    // Process spatial locations in parallel
    const int total_spatial = B * H_out * W_out;
    for (int spatial_idx = tid; spatial_idx < total_spatial; spatial_idx += num_threads) {
        const int w_out = spatial_idx % W_out;
        const int h_out = (spatial_idx / W_out) % H_out;
        const int b = spatial_idx / (W_out * H_out);
        
        // Calculate input position
        const int h_in = h_out * strideH - padH + kh * dilationH;
        const int w_in = w_out * strideW - padW + kw * dilationW;
        
        if (h_in < 0 || h_in >= H_in || w_in < 0 || w_in >= W_in) continue;
        
        // Load input values
        const int input_base = ((b * C_in_per_q + c_in) * Q) * H_in * W_in;
        const int input_offset = h_in * W_in + w_in;
        
        const scalar_t xr = input[input_base + 0 * H_in * W_in + input_offset];
        const scalar_t xi = input[input_base + 1 * H_in * W_in + input_offset];
        const scalar_t xj = input[input_base + 2 * H_in * W_in + input_offset];
        const scalar_t xk = input[input_base + 3 * H_in * W_in + input_offset];
        
        // Load gradient values
        const int grad_base = ((b * C_out_per_q + c_out) * Q) * H_out * W_out;
        const int grad_offset = h_out * W_out + w_out;
        
        const scalar_t grad_r = grad_output[grad_base + 0 * H_out * W_out + grad_offset];
        const scalar_t grad_i = grad_output[grad_base + 1 * H_out * W_out + grad_offset];
        const scalar_t grad_j = grad_output[grad_base + 2 * H_out * W_out + grad_offset];
        const scalar_t grad_k = grad_output[grad_base + 3 * H_out * W_out + grad_offset];
        
        // Accumulate weight gradients
        local_grad_r += (grad_r - grad_i - grad_j - grad_k) * xr;
        local_grad_i += (grad_r + grad_i + grad_j - grad_k) * xi;
        local_grad_j += (grad_r + grad_i + grad_j + grad_k) * xj;
        local_grad_k += (grad_r - grad_i + grad_j + grad_k) * xk;
    }
    
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        local_grad_r += __shfl_down_sync(0xffffffff, local_grad_r, offset);
        local_grad_i += __shfl_down_sync(0xffffffff, local_grad_i, offset);
        local_grad_j += __shfl_down_sync(0xffffffff, local_grad_j, offset);
        local_grad_k += __shfl_down_sync(0xffffffff, local_grad_k, offset);
    }
    
    // Block-level reduction using shared memory
    __shared__ acc_scalar_t shared_grads[4][32]; // Max 32 warps per block
    
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
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
        acc_scalar_t final_r = (lane_id < num_warps) ? shared_grads[0][lane_id] : 0;
        acc_scalar_t final_i = (lane_id < num_warps) ? shared_grads[1][lane_id] : 0;
        acc_scalar_t final_j = (lane_id < num_warps) ? shared_grads[2][lane_id] : 0;
        acc_scalar_t final_k = (lane_id < num_warps) ? shared_grads[3][lane_id] : 0;
        
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

// Simple and efficient bias gradient kernel
template <typename scalar_t>
__global__ void qconv_backward_bias_kernel(
    scalar_t* __restrict__ grad_bias,
    const scalar_t* __restrict__ grad_output,
    const int B,
    const int C_out_per_q,
    const int H_out,
    const int W_out) {
    
    using acc_scalar_t = typename std::conditional<std::is_same<scalar_t, double>::value, double, float>::type;
    
    const int c_out = blockIdx.x;
    if (c_out >= C_out_per_q) return;
    
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    acc_scalar_t local_sum = 0;
    
    // Sum over batch and spatial dimensions for the real component only
    const int total_elements = B * H_out * W_out;
    for (int idx = tid; idx < total_elements; idx += num_threads) {
        const int offset = ((idx / (H_out * W_out)) * C_out_per_q + c_out) * 4 * H_out * W_out + 
                          (idx % (H_out * W_out));
        local_sum += static_cast<acc_scalar_t>(grad_output[offset]);
    }
    
    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    // Block reduction
    __shared__ acc_scalar_t shared_sum[32];
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    if (lane_id == 0) {
        shared_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        const int num_warps = (num_threads + 31) / 32;
        acc_scalar_t final_sum = (lane_id < num_warps) ? shared_sum[lane_id] : 0;
        
        for (int offset = 16; offset > 0; offset /= 2) {
            final_sum += __shfl_down_sync(0xffffffff, final_sum, offset);
        }
        
        if (lane_id == 0) {
            grad_bias[c_out] = static_cast<scalar_t>(final_sum);
        }
    }
}

// Optimized backward function
std::vector<torch::Tensor> qconv_backward_cuda(
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
    const auto C_in_per_q = input.size(1);
    const auto H_in = input.size(3);
    const auto W_in = input.size(4);
    
    const auto C_out_per_q = weight_r.size(0);
    const auto C_in_per_q_grp = weight_r.size(1);
    const auto kH = weight_r.size(2);
    const auto kW = weight_r.size(3);
    
    const auto H_out = grad_output.size(3);
    const auto W_out = grad_output.size(4);
    
    // Pre-zero gradients to avoid initialization overhead
    auto grad_input = torch::zeros_like(input);
    auto grad_weight_r = torch::zeros_like(weight_r);
    auto grad_weight_i = torch::zeros_like(weight_i);
    auto grad_weight_j = torch::zeros_like(weight_j);
    auto grad_weight_k = torch::zeros_like(weight_k);
    auto grad_bias = bias_defined ? torch::zeros({C_out_per_q}, grad_output.options()) : torch::Tensor();
    
    // Ensure contiguous tensors
    grad_output = grad_output.contiguous();
    input = input.contiguous();
    weight_r = weight_r.contiguous();
    weight_i = weight_i.contiguous();
    weight_j = weight_j.contiguous();
    weight_k = weight_k.contiguous();
    
    // 1. Backward input - process all input elements
    {
        const int threads = 256;
        const int total_input_elements = B * C_in_per_q * H_in * W_in;
        const int blocks = (total_input_elements + threads - 1) / threads;
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "qconv_backward_input", ([&] {
            qconv_backward_input_kernel<scalar_t><<<blocks, threads>>>(
                grad_input.data_ptr<scalar_t>(),
                grad_output.data_ptr<scalar_t>(),
                weight_r.data_ptr<scalar_t>(),
                weight_i.data_ptr<scalar_t>(),
                weight_j.data_ptr<scalar_t>(),
                weight_k.data_ptr<scalar_t>(),
                B, C_in_per_q, C_out_per_q, H_in, W_in, H_out, W_out,
                kH, kW, stride[0], stride[1], padding[0], padding[1],
                dilation[0], dilation[1], groups
            );
        }));
    }
    
    // 2. Backward weight - one block per weight
    {
        const int threads = 256;
        const int total_weights = C_out_per_q * C_in_per_q_grp * kH * kW;
        const int blocks = total_weights;
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "qconv_backward_weight", ([&] {
            qconv_backward_weight_kernel<scalar_t><<<blocks, threads>>>(
                grad_weight_r.data_ptr<scalar_t>(),
                grad_weight_i.data_ptr<scalar_t>(),
                grad_weight_j.data_ptr<scalar_t>(),
                grad_weight_k.data_ptr<scalar_t>(),
                grad_output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                B, C_in_per_q, C_out_per_q, H_in, W_in, H_out, W_out,
                kH, kW, stride[0], stride[1], padding[0], padding[1],
                dilation[0], dilation[1], groups
            );
        }));
    }
    
    // 3. Backward bias - one block per output channel
    if (bias_defined) {
        const int threads = 256;
        const int blocks = C_out_per_q;
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "qconv_backward_bias", ([&] {
            qconv_backward_bias_kernel<scalar_t><<<blocks, threads>>>(
                grad_bias.data_ptr<scalar_t>(),
                grad_output.data_ptr<scalar_t>(),
                B, C_out_per_q, H_out, W_out
            );
        }));
    }
    
    return {grad_input, grad_weight_r, grad_weight_i, grad_weight_j, grad_weight_k, grad_bias};
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


    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "iqbn_forward_cuda", ([&] {
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
        // TORCH_CHECK(bias_i.defined() && bias_i.sizes() == bias_r.sizes(), "bias_i shape mismatch");
        // TORCH_CHECK(bias_j.defined() && bias_j.sizes() == bias_r.sizes(), "bias_j shape mismatch");
        // TORCH_CHECK(bias_k.defined() && bias_k.sizes() == bias_r.sizes(), "bias_k shape mismatch");
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


    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "qconv_forward_cuda_hamilton", ([&] {
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
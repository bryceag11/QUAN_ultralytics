// quaternion_ops.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void iqbn_forward_kernel(
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

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = B * C * HW;
    if (tid >= total_elements) return;

    // BCHWQ indexing
    // Decompose indices for coalesced access - consecutive threads access consecutive spatial locations
    const int hw_idx = tid % HW;
    const int c = (tid / HW) % C;
    const int b = tid / (HW * C);

    // Calculate base indices for quaternion components
    const int base_idx = tid * 4;
    const int stat_base_idx = c * 4;

    // Coalesced memory access using pointer arithmetic
    const scalar_t* input_ptr = &input[base_idx];
    const scalar_t* mean_ptr = &running_mean[stat_base_idx];
    const scalar_t* var_ptr = &running_var[stat_base_idx];
    const scalar_t* gamma_ptr = &gamma[stat_base_idx];
    const scalar_t* beta_ptr = &beta[stat_base_idx];
    scalar_t* output_ptr = &output[base_idx];

    // Process all 4 quaternion components with loop unrolling
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const scalar_t inp = input_ptr[q];
        const scalar_t mean_val = mean_ptr[q];
        const scalar_t var_val = var_ptr[q];
        const scalar_t gamma_val = gamma_ptr[q];
        const scalar_t beta_val = beta_ptr[q];

        output_ptr[q] = gamma_val * (inp - mean_val) / sqrt(var_val + eps) + beta_val;
    }
}


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
    
    using acc_scalar_t = typename std::conditional<std::is_same<scalar_t, double>::value, double, float>::type;

    const int total_outputs = B * C_out * H_out * W_out;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= total_outputs) return;

    // Decompose thread index
    const int w_out = tid % W_out;
    const int h_out = (tid / W_out) % H_out;
    const int c_out = (tid / (W_out * H_out)) % C_out;
    const int batch = tid / (W_out * H_out * C_out);

    const int C_in_grp = C_in / groups;
    const int group_id = c_out / (C_out / groups);
    const int c_in_start = group_id * C_in_grp;
    const int c_in_end = c_in_start + C_in_grp;

    acc_scalar_t sum_r_acc = bias_r ? static_cast<acc_scalar_t>(bias_r[c_out]) : 0.0;
    acc_scalar_t sum_i_acc = 0.0;
    acc_scalar_t sum_j_acc = 0.0;
    acc_scalar_t sum_k_acc = 0.0;

    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        for (int kh = 0; kh < kH; ++kh) {
            const int h_in = h_out * strideH - padH + kh * dilationH;
            if (h_in < 0 || h_in >= H_in) continue;
            
            for (int kw = 0; kw < kW; ++kw) {
                const int w_in = w_out * strideW - padW + kw * dilationW;
                if (w_in < 0 || w_in >= W_in) continue; 
                    const int weight_idx = ((c_out * C_in_grp + (c_in - c_in_start)) * kH + kh) * kW + kw;
                    
                    // Weights
                    const acc_scalar_t wr = static_cast<acc_scalar_t>(weight_r[weight_idx]);
                    const acc_scalar_t wi = static_cast<acc_scalar_t>(weight_i[weight_idx]);
                    const acc_scalar_t wj = static_cast<acc_scalar_t>(weight_j[weight_idx]);
                    const acc_scalar_t wk = static_cast<acc_scalar_t>(weight_k[weight_idx]);

                    // Input index for BCHWQ
                    const int input_base_idx = ((batch * C_in + c_in) * H_in + h_in) * W_in + w_in;
                    
                    // Inputs
                    const scalar_t* input_ptr = &input[input_base_idx * 4];
                    const acc_scalar_t xr = static_cast<acc_scalar_t>(input_ptr[0]);
                    const acc_scalar_t xi = static_cast<acc_scalar_t>(input_ptr[1]);
                    const acc_scalar_t xj = static_cast<acc_scalar_t>(input_ptr[2]);
                    const acc_scalar_t xk = static_cast<acc_scalar_t>(input_ptr[3]);

                    // Right separable
                    // sum_r_acc += static_cast<acc_scalar_t>(xr) * static_cast<acc_scalar_t>(wr);
                    // sum_i_acc += static_cast<acc_scalar_t>(xi) * static_cast<acc_scalar_t>(wi);
                    // sum_j_acc += static_cast<acc_scalar_t>(xj) * static_cast<acc_scalar_t>(wj);
                    // sum_k_acc += static_cast<acc_scalar_t>(xk) * static_cast<acc_scalar_t>(wk);

                    // Left separable
                    sum_r_acc += static_cast<acc_scalar_t>(wr) * static_cast<acc_scalar_t>(xr);
                    sum_i_acc += static_cast<acc_scalar_t>(wi) * static_cast<acc_scalar_t>(xi);
                    sum_j_acc += static_cast<acc_scalar_t>(wj) * static_cast<acc_scalar_t>(xj);
                    sum_k_acc += static_cast<acc_scalar_t>(wk) * static_cast<acc_scalar_t>(xk);

                    // Right Hamilton
                    // sum_r_acc += xr*wr - xi*wi - xj*wj - xk*wk;
                    // sum_i_acc += xr*wi + xi*wr + xj*wk - xk*wj;
                    // sum_j_acc += xr*wj - xi*wk + xj*wr + xk*wi;
                    // sum_k_acc += xr*wk + xi*wj - xj*wi + xk*wr;

                    // Left Hamilton
                    // sum_r_acc += wr*xr - wi*xi - wj*xj - wk*xk;
                    // sum_i_acc += wr*xi + wi*xr + wj*xk - wk*xj;
                    // sum_j_acc += wr*xj - wi*xk + wj*xr + wk*xi;
                    // sum_k_acc += wr*xk + wi*xj - wj*xi + wk*xr;
                }
            }
        }


    // Final Hamilton combination
    // acc_scalar_t final_r_acc = sum_r_acc;
    // acc_scalar_t final_i_acc = sum_i_acc;
    // acc_scalar_t final_j_acc = sum_j_acc;
    // acc_scalar_t final_k_acc = sum_k_acc;

    // Zhou separable forward calculation (CORRECTED)
    acc_scalar_t final_r_acc = sum_r_acc + sum_i_acc + sum_j_acc + sum_k_acc;
    acc_scalar_t final_i_acc = - sum_i_acc + sum_r_acc + sum_k_acc - sum_j_acc;
    acc_scalar_t final_j_acc = - sum_j_acc - sum_k_acc + sum_r_acc + sum_i_acc;
    acc_scalar_t final_k_acc = - sum_k_acc + sum_j_acc - sum_i_acc + sum_r_acc;

    // Zhou left separable forward calculation (INCORRECT)
    // acc_scalar_t final_r_acc = sum_r_acc + sum_i_acc + sum_j_acc + sum_k_acc;
    // acc_scalar_t final_i_acc = sum_i_acc - sum_r_acc - sum_k_acc + sum_j_acc;
    // acc_scalar_t final_j_acc = sum_j_acc + sum_k_acc - sum_r_acc - sum_i_acc;
    // acc_scalar_t final_k_acc = sum_k_acc - sum_j_acc + sum_i_acc - sum_r_acc;

    // Right separable
    // acc_scalar_t final_r_acc = sum_r_acc - sum_i_acc - sum_j_acc - sum_k_acc;
    // acc_scalar_t final_i_acc = sum_i_acc + sum_r_acc - sum_k_acc + sum_j_acc;
    // acc_scalar_t final_j_acc = sum_j_acc + sum_k_acc + sum_r_acc - sum_i_acc;
    // acc_scalar_t final_k_acc = sum_k_acc - sum_j_acc + sum_i_acc + sum_r_acc;

    // Left separable
    // acc_scalar_t final_r_acc = sum_r_acc - sum_i_acc - sum_j_acc - sum_k_acc;
    // acc_scalar_t final_i_acc = sum_i_acc + sum_r_acc + sum_k_acc - sum_j_acc;
    // acc_scalar_t final_j_acc = sum_j_acc - sum_k_acc + sum_r_acc + sum_i_acc;
    // acc_scalar_t final_k_acc = sum_k_acc + sum_j_acc - sum_i_acc + sum_r_acc;

    // BCHWQ output
    // Coalesced output write using pointer arithmetic
    scalar_t* output_ptr = &output[tid * 4];
    output_ptr[0] = static_cast<scalar_t>(final_r_acc);
    output_ptr[1] = static_cast<scalar_t>(final_i_acc);
    output_ptr[2] = static_cast<scalar_t>(final_j_acc);
    output_ptr[3] = static_cast<scalar_t>(final_k_acc);
}


template <typename scalar_t>
__global__ void qconv_backward_input_kernel(
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

    using acc_scalar_t = typename std::conditional<std::is_same<scalar_t, double>::value, double, float>::type;
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = B * C_in * H_in * W_in;
    
    if (tid >= total_elements) return;
    
    // Decompose indices for coalesced access
    const int w_in = tid % W_in;
    const int h_in = (tid / W_in) % H_in;
    const int c_in = (tid / (W_in * H_in)) % C_in;
    const int batch = tid / (W_in * H_in * C_in);
    
    const int group = c_in / (C_in / groups);
    const int C_out_grp = C_out / groups;
    const int c_out_start = group * C_out_grp;
    const int c_out_end = c_out_start + C_out_grp;
    const int C_in_grp = C_in / groups;
    
    acc_scalar_t grad_xr = 0, grad_xi = 0, grad_xj = 0, grad_xk = 0;
    
    // Process each output channel
    for (int c_out = c_out_start; c_out < c_out_end; ++c_out) {
        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                const int h_out = (h_in + padH - kh * dilationH);
                const int w_out = (w_in + padW - kw * dilationW);
                
                // Early exit for invalid output positions - reduces thread divergence
                if (h_out % strideH != 0 || w_out % strideW != 0) continue;
                
                const int h_out_idx = h_out / strideH;
                const int w_out_idx = w_out / strideW;
                
                if (h_out_idx < 0 || h_out_idx >= H_out || w_out_idx < 0 || w_out_idx >= W_out) continue;

                // Coalesced gradient output access using pointer arithmetic
                const int grad_out_base_idx = ((batch * C_out + c_out) * H_out + h_out_idx) * W_out + w_out_idx;
                const scalar_t* grad_out_ptr = &grad_output[grad_out_base_idx * 4];
                
                const acc_scalar_t grad_r = static_cast<acc_scalar_t>(grad_out_ptr[0]);
                const acc_scalar_t grad_i = static_cast<acc_scalar_t>(grad_out_ptr[1]);
                const acc_scalar_t grad_j = static_cast<acc_scalar_t>(grad_out_ptr[2]);
                const acc_scalar_t grad_k = static_cast<acc_scalar_t>(grad_out_ptr[3]);
                
                const int weight_idx = ((c_out * C_in_grp + (c_in - group * C_in_grp)) * kH + kh) * kW + kw;
                
                const acc_scalar_t wr = static_cast<acc_scalar_t>(weight_r[weight_idx]);
                const acc_scalar_t wi = static_cast<acc_scalar_t>(weight_i[weight_idx]);
                const acc_scalar_t wj = static_cast<acc_scalar_t>(weight_j[weight_idx]);
                const acc_scalar_t wk = static_cast<acc_scalar_t>(weight_k[weight_idx]);
                

                        // Left separable
                        // const acc_scalar_t combined_grad_for_r = grad_r + grad_i + grad_j + grad_k;
                        // const acc_scalar_t combined_grad_for_i = grad_i - grad_r - grad_k + grad_j;
                        // const acc_scalar_t combined_grad_for_j = grad_j + grad_k - grad_r - grad_i;
                        // const acc_scalar_t combined_grad_for_k = grad_k - grad_j + grad_i - grad_r;                
                    
                        // Right separable 
                        // const acc_scalar_t combined_grad_for_r = grad_r + grad_i + grad_j + grad_k;
                        // const acc_scalar_t combined_grad_for_i = grad_i - grad_r + grad_k - grad_j;
                        // const acc_scalar_t combined_grad_for_j = grad_j - grad_k - grad_r + grad_i;
                        // const acc_scalar_t combined_grad_for_k = grad_k + grad_j - grad_i - grad_r;



                        // Left Conj separable
                        // const acc_scalar_t combined_grad_for_r = grad_r - grad_i - grad_j - grad_k;
                        // const acc_scalar_t combined_grad_for_i = grad_i + grad_r + grad_k - grad_j;
                        // const acc_scalar_t combined_grad_for_j = grad_j - grad_k + grad_r + grad_i;
                        // const acc_scalar_t combined_grad_for_k = grad_k + grad_j - grad_i + grad_r;                
                        
                        // Correct Left Conj separable 
                        grad_xr += (grad_r + grad_i + grad_j + grad_k)  *wr;
                        grad_xi += (- grad_i + grad_r - grad_k + grad_j)*wi;
                        grad_xj += (- grad_j + grad_k + grad_r - grad_i)*wj;
                        grad_xk += (- grad_k - grad_j + grad_i + grad_r)*wk;

                        // Combination
                        // grad_xr += combined_grad_for_r * wr;
                        // grad_xi += combined_grad_for_i * wi;
                        // grad_xj += combined_grad_for_j * wj;
                        // grad_xk += combined_grad_for_k * wk;

                        // Hamilton Left Mult (w*x)
                        // grad_xr += (grad_r * wr + grad_i * wi + grad_j * wj + grad_k * wk);
                        // grad_xi += (- grad_r * wi + grad_i * wr + grad_j * wk - grad_k * wj);
                        // grad_xj += (- grad_r * wj - grad_i * wk + grad_j * wr + grad_k * wi);
                        // grad_xk += (- grad_r * wk + grad_i * wj - grad_j * wi + grad_k * wr); 

                        // Hamilton Right Mult  (x*w)
                        // grad_xr += (grad_r * wr + grad_i * wi + grad_j * wj + grad_k * wk);
                        // grad_xi += (- grad_r * wi + grad_i * wr - grad_j * wk + grad_k * wj);
                        // grad_xj += (- grad_r * wj + grad_i * wk + grad_j * wr - grad_k * wi);
                        // grad_xk += (- grad_r * wk - grad_i * wj + grad_j * wi + grad_k * wr); 
                    }
                }
            }

    // BCHWQ output
    // Coalesced output write using pointer arithmetic
    scalar_t* grad_input_ptr = &grad_input[tid * 4];
    grad_input_ptr[0] = static_cast<scalar_t>(grad_xr);
    grad_input_ptr[1] = static_cast<scalar_t>(grad_xi);
    grad_input_ptr[2] = static_cast<scalar_t>(grad_xj);
    grad_input_ptr[3] = static_cast<scalar_t>(grad_xk);
}

template <typename scalar_t>
__global__ void qconv_backward_weight_kernel(
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

    using acc_scalar_t = typename std::conditional<std::is_same<scalar_t, double>::value, double, float>::type;
    
    const int weight_idx = blockIdx.x;
    const int C_in_grp = C_in / groups;
    const int total_weights = C_out * C_in_grp * kH * kW;
    
    if (weight_idx >= total_weights) return;
    
    const int kw = weight_idx % kW;
    const int kh = (weight_idx / kW) % kH;
    const int c_in = (weight_idx / (kW * kH)) % C_in_grp;
    const int c_out = weight_idx / (kW * kH * C_in_grp);
    
    const int group = c_out / (C_out / groups);
    const int global_c_in = group * C_in_grp + c_in;
    
    __shared__ acc_scalar_t shared_grads[4][32];
    
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    acc_scalar_t local_grad_r = 0, local_grad_i = 0, local_grad_j = 0, local_grad_k = 0;
    
    const int total_pixels = B * H_out * W_out;

    for (int idx = tid; idx < total_pixels; idx += num_threads) {
        const int w_out = idx % W_out;
        const int h_out = (idx / W_out) % H_out;
        const int batch = idx / (W_out * H_out);
        
        const int h_in = h_out * strideH - padH + kh * dilationH;
        const int w_in = w_out * strideW - padW + kw * dilationW;
        
        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
            // Grad output index for BCHWQ layout
            const int grad_out_base_idx = ((batch * C_out + c_out) * H_out + h_out) * W_out + w_out;
            
            const acc_scalar_t grad_r = static_cast<acc_scalar_t>(grad_output[grad_out_base_idx * 4 + 0]);
            const acc_scalar_t grad_i = static_cast<acc_scalar_t>(grad_output[grad_out_base_idx * 4 + 1]);
            const acc_scalar_t grad_j = static_cast<acc_scalar_t>(grad_output[grad_out_base_idx * 4 + 2]);
            const acc_scalar_t grad_k = static_cast<acc_scalar_t>(grad_output[grad_out_base_idx * 4 + 3]);
            
            // Input index for BCHWQ layout
            const int input_base_idx = ((batch * C_in + global_c_in) * H_in + h_in) * W_in + w_in;
            
            const acc_scalar_t xr = static_cast<acc_scalar_t>(input[input_base_idx * 4 + 0]);
            const acc_scalar_t xi = static_cast<acc_scalar_t>(input[input_base_idx * 4 + 1]);
            const acc_scalar_t xj = static_cast<acc_scalar_t>(input[input_base_idx * 4 + 2]);
            const acc_scalar_t xk = static_cast<acc_scalar_t>(input[input_base_idx * 4 + 3]);

            // Left Conj separable
            // local_grad_r += (grad_r - grad_i - grad_j - grad_k) * xr;
            // local_grad_i += (grad_i + grad_r + grad_k - grad_j) * xi;
            // local_grad_j += (grad_j - grad_k + grad_r + grad_i) * xj;
            // local_grad_k += (grad_k + grad_j - grad_i + grad_r) * xk;                
            
            // Correct Left Conj separable 
            local_grad_r += (grad_r + grad_i + grad_j + grad_k) * xr;
            local_grad_i += (- grad_i + grad_r - grad_k + grad_j) * xi;
            local_grad_j += (- grad_j + grad_k + grad_r - grad_i) * xj;
            local_grad_k += (- grad_k - grad_j + grad_i + grad_r) * xk;

            // Left separable 
            // local_grad_r += (grad_r + grad_i + grad_j + grad_k) * xr;
            // local_grad_i += (grad_i - grad_r - grad_k + grad_j) * xi;
            // local_grad_j += (grad_j + grad_k - grad_r - grad_i) * xj;
            // local_grad_k += (grad_k - grad_j + grad_i - grad_r) * xk;
            
            // Right separable
            // local_grad_r += (grad_r + grad_i + grad_j + grad_k) * xr;
            // local_grad_i += (grad_i - grad_r + grad_k - grad_j) * xi;
            // local_grad_j += (grad_j - grad_k - grad_r + grad_i) * xj;
            // local_grad_k += (grad_k + grad_j - grad_i - grad_r) * xk;

            // Hamilton left mult
            // local_grad_r += (grad_r * xr + grad_i * xi + grad_j * xj + grad_k * xk);
            // local_grad_i += (- grad_r * xi + grad_i * xr - grad_j * xk + grad_k * xj);
            // local_grad_j += (- grad_r * xj + grad_i * xk + grad_j * xr - grad_k * xi);
            // local_grad_k += (- grad_r * xk - grad_i * xj + grad_j * xi + grad_k * xr);

            // Hamilton right mult 
            // local_grad_r += (grad_r * xr + grad_i * xi + grad_j * xj + grad_k * xk);
            // local_grad_i += (- grad_r * xi + grad_i * xr + grad_j * xk - grad_k * xj);
            // local_grad_j += (- grad_r * xj - grad_i * xk + grad_j * xr + grad_k * xi);
            // local_grad_k += (- grad_r * xk + grad_i * xj - grad_j * xi + grad_k * xr);
        }

    }
    
    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        local_grad_r += __shfl_down_sync(0xffffffff, local_grad_r, offset);
        local_grad_i += __shfl_down_sync(0xffffffff, local_grad_i, offset);
        local_grad_j += __shfl_down_sync(0xffffffff, local_grad_j, offset);
        local_grad_k += __shfl_down_sync(0xffffffff, local_grad_k, offset);
    }
    

    
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    if (lane_id == 0) {
        shared_grads[0][warp_id] = local_grad_r;
        shared_grads[1][warp_id] = local_grad_i;
        shared_grads[2][warp_id] = local_grad_j;
        shared_grads[3][warp_id] = local_grad_k;
    }
    __syncthreads();
    
    // Final reduction
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

template <typename scalar_t>
__global__ void qconv_backward_bias_kernel(
    scalar_t* __restrict__ grad_bias,
    const scalar_t* __restrict__ grad_output,
    const int B,
    const int C_out,
    const int H_out,
    const int W_out) {
    
    using acc_scalar_t = typename std::conditional<std::is_same<scalar_t, double>::value, double, float>::type;
    
    const int c_out = blockIdx.x;
    if (c_out >= C_out) return;
    
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    acc_scalar_t local_sum = 0;
    
    // Sum over batch and spatial dimensions for the real component only
    const int total_elements = B * H_out * W_out;
    for (int idx = tid; idx < total_elements; idx += num_threads) {
        const int w_out = idx % W_out;
        const int h_out = (idx / W_out) % H_out;
        const int batch = idx / (H_out * W_out);
        
        // BCHWQ output
        const int grad_out_idx = ((batch * C_out + c_out) * H_out + h_out) * W_out + w_out;
        local_sum += static_cast<acc_scalar_t>(grad_output[grad_out_idx * 4 + 0]);
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
    const auto C_in = input.size(1);
    const auto H_in = input.size(2);
    const auto W_in = input.size(3);
    
    const auto C_out = weight_r.size(0);
    const auto C_in_grp = weight_r.size(1);
    const auto kH = weight_r.size(2);
    const auto kW = weight_r.size(3);
    
    const auto H_out = grad_output.size(2);
    const auto W_out = grad_output.size(3);
    
    // Pre-zero gradients to avoid initialization overhead
    auto grad_input = torch::zeros_like(input);
    auto grad_weight_r = torch::zeros_like(weight_r);
    auto grad_weight_i = torch::zeros_like(weight_i);
    auto grad_weight_j = torch::zeros_like(weight_j);
    auto grad_weight_k = torch::zeros_like(weight_k);
    auto grad_bias = bias_defined ? torch::zeros({C_out}, grad_output.options()) : torch::Tensor();
    
    // Ensure contiguous tensors
    // grad_output = grad_output.contiguous();
    // input = input.contiguous();
    // weight_r = weight_r.contiguous();
    // weight_i = weight_i.contiguous();
    // weight_j = weight_j.contiguous();
    // weight_k = weight_k.contiguous();
    
    // Backward input
//     {
//         const int threads = 256;
//         const int total_input_elements = B * C_in_per_q * H_in * W_in;
//         const int blocks = (total_input_elements + threads - 1) / threads;
        
//         AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "qconv_backward_input", ([&] {
//             qconv_backward_input_kernel<scalar_t><<<blocks, threads>>>(
//                 grad_input.data_ptr<scalar_t>(),
//                 grad_output.data_ptr<scalar_t>(),
//                 weight_r.data_ptr<scalar_t>(),
//                 weight_i.data_ptr<scalar_t>(),
//                 weight_j.data_ptr<scalar_t>(),
//                 weight_k.data_ptr<scalar_t>(),
//                 B, C_in_per_q, C_out_per_q, H_in, W_in, H_out, W_out,
//                 kH, kW, stride[0], stride[1], padding[0], padding[1],
//                 dilation[0], dilation[1], groups
//             );
//         }));
//     }
    
//     // Backward weight has one block per weight
//     {
//         const int threads = 256;
//         const int total_weights = C_out_per_q * C_in_per_q_grp * kH * kW;
//         const int blocks = total_weights;
        
//         AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "qconv_backward_weight", ([&] {
//             qconv_backward_weight_kernel<scalar_t><<<blocks, threads>>>(
//                 grad_weight_r.data_ptr<scalar_t>(),
//                 grad_weight_i.data_ptr<scalar_t>(),
//                 grad_weight_j.data_ptr<scalar_t>(),
//                 grad_weight_k.data_ptr<scalar_t>(),
//                 grad_output.data_ptr<scalar_t>(),
//                 input.data_ptr<scalar_t>(),
//                 B, C_in_per_q, C_out_per_q, H_in, W_in, H_out, W_out,
//                 kH, kW, stride[0], stride[1], padding[0], padding[1],
//                 dilation[0], dilation[1], groups
//             );
//         }));
//     }
    
//     // Backward bias has one block per output channel
//     if (bias_defined) {
//         const int threads = 256;
//         const int blocks = C_out_per_q;
        
//         AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "qconv_backward_bias", ([&] {
//             qconv_backward_bias_kernel<scalar_t><<<blocks, threads>>>(
//                 grad_bias.data_ptr<scalar_t>(),
//                 grad_output.data_ptr<scalar_t>(),
//                 B, C_out_per_q, H_out, W_out
//             );
//         }));
//     }
    
//     return {grad_input, grad_weight_r, grad_weight_i, grad_weight_j, grad_weight_k, grad_bias};
// }
    const int threads = 512;
    
    // Gradient w.r.t. input
    const int input_blocks = (B * C_in * H_in * W_in + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "qconv_backward_input", ([&] {
        qconv_backward_input_kernel<scalar_t><<<input_blocks, threads>>>(
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
    
    // Gradient w.r.t. weights
    const int weight_blocks = C_out * C_in_grp * kH * kW;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "qconv_backward_weight", ([&] {
        qconv_backward_weight_kernel<scalar_t><<<weight_blocks, threads>>>(
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
    
    // Gradient w.r.t. bias
    if (bias_defined) {
        const int bias_threads = 512;
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "qconv_backward_bias", ([&] {
            qconv_backward_bias_kernel<scalar_t><<<C_out, bias_threads>>>(
                grad_bias.data_ptr<scalar_t>(),
                grad_output.data_ptr<scalar_t>(),
                B, C_out, H_out, W_out
            );
        }));
    }
    
    return {grad_input, grad_weight_r, grad_weight_i, grad_weight_j, grad_weight_k, grad_bias};
}



torch::Tensor iqbn_forward_cuda(
    torch::Tensor input,        // [B, C, H, W, Q]
    torch::Tensor gamma,        // [C, Q=4]
    torch::Tensor beta,         // [C, Q=4]
    torch::Tensor running_mean, // [C, Q=4]
    torch::Tensor running_var,  // [C, Q=4]
    float eps) {

    TORCH_CHECK(input.dim() == 5, "iqbn_forward_cuda expects 5D input");
    TORCH_CHECK(input.size(4) == 4, "Input quaternion dimension (dim 4) must be 4");
    
    const auto B = input.size(0);
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);
    const auto HW = H * W;

    // Check parameter shapes
    TORCH_CHECK(gamma.size(0) == C && gamma.size(1) == 4, "gamma shape mismatch");
    TORCH_CHECK(beta.size(0) == C && beta.size(1) == 4, "beta shape mismatch");
    TORCH_CHECK(running_mean.size(0) == C && running_mean.size(1) == 4, "running_mean shape mismatch");
    TORCH_CHECK(running_var.size(0) == C && running_var.size(1) == 4, "running_var shape mismatch");

    auto output = torch::empty_like(input);

    const int threads = 512;
    const int total_elements = B * C * HW;
    const int blocks = (total_elements + threads - 1) / threads;

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
            B, C, HW, eps
        );
    }));

    return output;
}



torch::Tensor qconv_forward_cuda(
    torch::Tensor input,        // [B, C_in, H_in, W_in, Q=4]
    torch::Tensor weight_r,     // [C_out, C_in_grp, kH, kW]
    torch::Tensor weight_i,     // [C_out, C_in_grp, kH, kW]
    torch::Tensor weight_j,     // [C_out, C_in_grp, kH, kW]
    torch::Tensor weight_k,     // [C_out, C_in_grp, kH, kW]
    torch::Tensor bias_r,       // [C_out] or None
    torch::Tensor bias_i,       // [C_out] or None
    torch::Tensor bias_j,       // [C_out] or None
    torch::Tensor bias_k,       // [C_out] or None
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups) {

    TORCH_CHECK(input.dim() == 5, "qconv_forward_cuda expects 5D input");
    TORCH_CHECK(input.size(4) == 4, "Input quaternion dimension (dim 4) must be 4");
    TORCH_CHECK(weight_r.dim() == 4, "weight_r must be 4D");
    TORCH_CHECK(weight_i.dim() == 4, "weight_i must be 4D");
    TORCH_CHECK(weight_j.dim() == 4, "weight_j must be 4D");
    TORCH_CHECK(weight_k.dim() == 4, "weight_k must be 4D");

    const auto B = input.size(0);
    const auto C_in = input.size(1);
    const auto H_in = input.size(2);
    const auto W_in = input.size(3);

    const auto C_out = weight_r.size(0);
    const auto C_in_grp = weight_r.size(1);
    const auto kH = weight_r.size(2);
    const auto kW = weight_r.size(3);

    // Basic shape consistency checks
    TORCH_CHECK(C_in == C_in_grp * groups, "Input channels must equal C_in_grp * groups");

    // Calculate output dimensions
    const int H_out = (H_in + 2 * padding[0] - dilation[0] * (kH - 1) - 1) / stride[0] + 1;
    const int W_out = (W_in + 2 * padding[1] - dilation[1] * (kW - 1) - 1) / stride[1] + 1;

    // Create output tensor with BCHWQ layout
    auto output = torch::empty({B, C_out, H_out, W_out, 4}, input.options());

    const int threads = 512;
    const int blocks = (B * C_out * H_out * W_out + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "qconv_forward_cuda", ([&] {
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
            B, C_in, C_out, H_in, W_in, H_out, W_out,
            kH, kW, stride[0], stride[1], padding[0], padding[1],
            dilation[0], dilation[1], groups
        );
    }));

    return output;
}
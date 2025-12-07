# quaternion_autograd_cuda.py - Updated autograd function
import torch
from torch.autograd import Function

try:
    import quaternion_ops
    CUDA_EXT = True
except ImportError:
    print("Failed to import quaternion_ops CUDA extension")
    CUDA_EXT = False

class QConvFunction(Function):
    @torch.amp.custom_fwd(cast_inputs=torch.float16, device_type='cuda')
    @staticmethod
    def forward(ctx, input, weight_r, weight_i, weight_j, weight_k, 
                bias_r, stride, padding, dilation, groups, input_shape_original=None):
        # Save tensors needed for backward
        ctx.save_for_backward(input, weight_r, weight_i, weight_j, weight_k)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.bias_defined = bias_r is not None
        ctx.input_shape_original = input_shape_original
        
        # Call optimized CUDA forward
        output = quaternion_ops.qconv_forward(
            input, weight_r, weight_i, weight_j, weight_k,
            bias_r, None, None, None,  # Only bias_r is used
            stride, padding, dilation, groups
        )
        return output

    @torch.amp.custom_bwd(device_type='cuda')
    @staticmethod
    def backward(ctx, grad_output):
        input, weight_r, weight_i, weight_j, weight_k = ctx.saved_tensors
        
        # Call optimized CUDA backward
        grad_input, grad_weight_r, grad_weight_i, grad_weight_j, grad_weight_k, grad_bias_r = \
            quaternion_ops.qconv_backward(
                grad_output.contiguous(),
                input.contiguous(),
                weight_r.contiguous(),
                weight_i.contiguous(),
                weight_j.contiguous(),
                weight_k.contiguous(),
                ctx.bias_defined,
                ctx.stride,
                ctx.padding,
                ctx.dilation,
                ctx.groups
            )
        
        # If input was reshaped in forward, reshape gradient back to original shape
        if ctx.input_shape_original is not None:
            grad_input = grad_input.reshape(ctx.input_shape_original)
        
        # Return gradients for all inputs (None for non-tensor inputs)
        return grad_input, grad_weight_r, grad_weight_i, grad_weight_j, grad_weight_k, \
               grad_bias_r, None, None, None, None, None



# Create a convenient function to use
def qconv2d_function(input, weight_r, weight_i, weight_j, weight_k, bias_r,
                        stride, padding, dilation, groups, input_shape_original=None):
    return QConvFunction.apply(input, weight_r, weight_i, weight_j, weight_k,
                              bias_r, stride, padding, dilation, groups, input_shape_original)
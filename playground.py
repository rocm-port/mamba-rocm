import torch
import torch.nn.functional as F
from einops import rearrange



def compare_tensor(out, out_ref, rtol=None, atol=None, verbose=False, name="output", raise_err=True):
    """
    Compare two torch tensors and assert if they are equal within a tolerance.

    Args:
        out (torch.Tensor): The output tensor to be compared.
        out_ref (torch.Tensor): The reference output tensor.
        rtol (float, optional): The relative tolerance. Defaults to 1e-5.
        atol (float, optional): The absolute tolerance. Defaults to 1e-8.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    """
    if verbose:
        print("out shape", out.shape, out_ref.shape)
    if rtol is None and atol is None:
        rtol, atol = (6e-4, 2e-3)
    assert(out.shape == out_ref.shape)

    if verbose:
        print(f'{name} max diff: {(out - out_ref).abs().max().item()}')
        print(f'{name} mean diff: {(out - out_ref).abs().mean().item()}')
    if raise_err:
        assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)
    else:
        print("passed test", torch.allclose(out, out_ref, rtol=rtol, atol=atol))

def print_memory_layout(tensor):
    strides = tensor.stride()
    dimensions = list(range(tensor.dim()))
    # Sort dimensions based on strides in descending order (largest stride first)
    sorted_dims = sorted(dimensions, key=lambda x: strides[x], reverse=True)
    layout = "->".join(f"Dim {dim} (Stride {strides[dim]})" for dim in sorted_dims)
    print("Memory layout order:", layout)

def inspect_tensor_properties(tensor):
    """
    Inspect and print various properties of a given PyTorch tensor.
    
    Args:
        tensor (torch.Tensor): The tensor to inspect.
    """
    print("Shape:", tensor.shape)
    print("Data type:", tensor.dtype)
    print("Device:", tensor.device)
    print("Requires grad:", tensor.requires_grad)
    print("Is contiguous:", tensor.is_contiguous())
    print("Memory format:", end=" ")
    print_memory_layout(tensor)
    print("Contains NaN:", torch.isnan(tensor).any().item())
    print("Contains Inf:", torch.isinf(tensor).any().item())
    print("Sample values:", tensor.flatten()[:10])
    print("Minimum value:", torch.min(tensor).item())
    print("Maximum value:", torch.max(tensor).item())
    print("Mean value:", torch.mean(tensor.float()).item())  # Use float() if necessary to avoid dtype issues
    print("Standard deviation:", torch.std(tensor.float()).item())  # Adding standard deviation



# Define input dimensions
b = 2  # batch size
d = 768  # feature dimension
l = 128  # sequence length

device = "cuda"

# # Create a tensor of shape (b, d, l)
y = 100 + 8000*torch.randn(b, d, l, device=device)
# y = -30000 + 100000*torch.randn(b*l, d, device=device)

# print("Original tensor shape:", y.shape)

# # Ensure the tensor is contiguous
# assert y.is_contiguous(), "Tensor is not contiguous"

# Rearrange the tensor to shape (b, l, d) and make it non-contiguous
y_rearranged = rearrange(y, 'b d l -> b l d')
# y_rearranged = y
# y_rearranged = torch.load("../culprit_tensor.pth")

print("_"*50)
inspect_tensor_properties(y_rearranged)
print("_"*50)

# torch.save(y_rearranged, "../temp.pth")
# y_rearranged = torch.load("../temp.pth")
print("Rearranged tensor layout:", end=" ")
print_memory_layout(y_rearranged)

# print(y_rearranged.requires_grad)
# y_rearranged = y_rearranged.requires_grad_()
# print(y_rearranged.requires_grad)

temp_cpu = y_rearranged.clone().to('cpu')

# Define linear layer parameters
out_proj_weight = torch.randn(d // 2, d, device=device, requires_grad=True)
# out_proj_weight = torch.load("../out_proj_weight.pth")
print(out_proj_weight.requires_grad)

out_proj_weight_cpu = out_proj_weight.clone().to('cpu')

# Apply the linear transformation to the non-contiguous tensor
result = F.linear(y_rearranged, out_proj_weight)


# # Make a contiguous copy of the rearranged tensor
# y_rearranged_contiguous = y_rearranged.contiguous()
# print("Rearranged contiguous tensor layout:", end=" ")
# print_memory_layout(y_rearranged_contiguous)

# out_proj_weight_copy = out_proj_weight.clone() # requires grad attribute is not copied

# # Apply the linear transformation to the contiguous tensor
# result_contiguous = F.linear(y_rearranged_contiguous, out_proj_weight_copy)


# Perform the linear operation on the CPU
result_cpu = F.linear(temp_cpu, out_proj_weight_cpu)

# Move the result back to the GPU
result_cpu_gpu = result_cpu.to('cuda')

# Compare the results
compare_tensor(result, result_cpu_gpu, verbose=True, raise_err=False)



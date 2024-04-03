
import math

import torch
import torch.nn.functional as F
import pytest

from einops import rearrange

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, mamba_inner_ref

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # TODO - remove

is_variable_B = True
is_variable_C = True
varBC_groups = True
has_D = True
has_z = True
has_delta_bias = True
delta_softplus = True
return_last_state = True
seqlen = 256 #256
itype = torch.float32
wtype = torch.float32

if varBC_groups > 1 and (not is_variable_B or not is_variable_C):
    pytest.skip()  # This config is not applicable
device = 'cuda'
rtol, atol = (6e-4, 2e-3) if itype == torch.float32 else (3e-3, 5e-3)
if itype == torch.bfloat16:
    rtol, atol = 3e-2, 5e-2
rtolw, atolw = (1e-3, 1e-3)
if has_z:  # If we have z, the errors on the weights seem higher
    rtolw = max(rtolw, rtol)
    atolw = max(atolw, atol)
# set seed
torch.random.manual_seed(0)
batch_size = 2
dim = 4
dstate = 8
is_complex = wtype == torch.complex64
A = (-0.5 * torch.rand(dim, dstate, device=device, dtype=wtype)).requires_grad_()
if not is_variable_B:
    B_shape = (dim, dstate)
elif varBC_groups == 1:
    B_shape = (batch_size, dstate, seqlen if not is_complex else seqlen * 2)
else:
    B_shape = (batch_size, varBC_groups, dstate, seqlen if not is_complex else seqlen * 2)
B = torch.randn(*B_shape, device=device, dtype=wtype if not is_variable_B else itype,
                requires_grad=True)
if not is_variable_C:
    C_shape = (dim, dstate)
elif varBC_groups == 1:
    C_shape = (batch_size, dstate, seqlen if not is_complex else seqlen * 2)
else:
    C_shape = (batch_size, varBC_groups, dstate, seqlen if not is_complex else seqlen * 2)
C = torch.randn(*C_shape, device=device, dtype=wtype if not is_variable_C else itype,
                requires_grad=True)
if has_D:
    D = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
else:
    D = None
if has_z:
    z = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype, requires_grad=True)
else:
    z = None
if has_delta_bias:
    delta_bias = (0.5 * torch.rand(dim, device=device, dtype=torch.float32)).requires_grad_()
else:
    delta_bias = None
u = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype, requires_grad=True)
delta = (0.5 * torch.rand(batch_size, dim, seqlen, device=device, dtype=itype)).requires_grad_()
A_ref = A.detach().clone().requires_grad_()
B_ref = B.detach().clone().requires_grad_()
C_ref = C.detach().clone().requires_grad_()
D_ref = D.detach().clone().requires_grad_() if D is not None else None
z_ref = z.detach().clone().requires_grad_() if z is not None else None
u_ref = u.detach().clone().requires_grad_()
delta_ref = delta.detach().clone().requires_grad_()
delta_bias_ref = delta_bias.detach().clone().requires_grad_() if delta_bias is not None else None
out, *rest = selective_scan_fn(
    u, delta, A, B, C, D, z=z,
    delta_bias=delta_bias, delta_softplus=delta_softplus,
    return_last_state=return_last_state
)
if return_last_state:
    state = rest[0]


## TODO: uncomment, to test the strange interference situation
# out_ref, *rest = selective_scan_ref(
#     u_ref, delta_ref, A_ref, B_ref, C_ref, D_ref, z=z_ref,
#     delta_bias=delta_bias_ref, delta_softplus=delta_softplus,
#     return_last_state=return_last_state
# )
# if return_last_state:
#     state_ref = rest[0]



print("Completed the forward launch", flush=True)


torch.cuda.synchronize(device=None)
print("Synced", flush=True)

#torch.cuda.set_sync_debug_mode(1)

g = torch.randn_like(out)
#out_ref.backward(g)
out.backward(g)

print("Completed the backward launch", flush=True)

torch.cuda.synchronize(device=None)
print("Synced", flush=True)

# tmp = u.grad
# print(tmp)
# print(f"U grad: {u.grad}")

# print(f"Delta grad: {delta.grad}")

# print(f'du max diff: {(u.grad - u_ref.grad).abs().max().item()}')
# print(f'ddelta max diff: {(delta.grad - delta_ref.grad).abs().max().item()}')
# print(f'dA max diff: {(A.grad - A_ref.grad).abs().max().item()}')
# print(f'dB max diff: {(B.grad - B_ref.grad).abs().max().item()}')
# print(f'dC max diff: {(C.grad - C_ref.grad).abs().max().item()}')
# if has_D:
#     print(f'dD max diff: {(D.grad - D_ref.grad).abs().max().item()}')
# if has_z:
#     print(f'dz max diff: {(z.grad - z_ref.grad).abs().max().item()}')
# if has_delta_bias:
#     print(f'ddelta_bias max diff: {(delta_bias.grad - delta_bias_ref.grad).abs().max().item()}')

# assert torch.allclose(u.grad, u_ref.grad.to(dtype=itype), rtol=rtol * 2, atol=atol * 2)
# assert torch.allclose(delta.grad, delta_ref.grad.to(dtype=itype), rtol=rtol * 5, atol=atol * 10)
# assert torch.allclose(A.grad, A_ref.grad, rtol=rtolw, atol=atolw * 5)
# assert torch.allclose(B.grad, B_ref.grad, rtol=rtolw if not is_variable_B else rtol,
#                         atol=atolw if not is_variable_B else atol)
# assert torch.allclose(C.grad, C_ref.grad, rtol=rtolw if not is_variable_C else rtol,
#                         atol=atolw if not is_variable_C else atol)
# if has_D:
#     assert torch.allclose(D.grad, D_ref.grad, rtol=rtolw, atol=atolw)
# if has_z:
#     assert torch.allclose(z.grad, z_ref.grad, rtol=rtolw, atol=atolw)
# if has_delta_bias:
#     assert torch.allclose(delta_bias.grad, delta_bias_ref.grad, rtol=rtolw, atol=atolw)

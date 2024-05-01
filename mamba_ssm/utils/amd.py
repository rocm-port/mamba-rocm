import torch
import torch.nn as nn

def hip_optimize_linear(model):
    """
        adjust some Linear layers' weights in Mamba to be contiguous over K dim to maximize performance with the amd instinct gpus
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        properties = torch.cuda.get_device_properties(device)
        name = properties.name.lower()
        if 'amd instinct' in name:
            for name, module in model.named_children():
                if isinstance(module, nn.Linear) and name in ['out_proj', 'x_proj']:
                    w = module.weight.t()
                    w = w.contiguous()
                    module.weight = torch.nn.Parameter(w.t())
                elif isinstance(module, nn.Module):
                    hip_optimize_linear(module)
    return model


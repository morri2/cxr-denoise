import torch
import torch.nn as nn
import numpy as np
from ptflops import get_model_complexity_info


def model_stats(model: nn.Module, input_size=(1, 512, 512), device='cpu'):
    model = model.to(device)

    # Size of state_dict in MB
    param_size = sum(p.numel() * p.element_size() for p in model.state_dict().values())
    param_size_MB = param_size / (1024 ** 2)

    # Compute FLOPs and params using ptflops
    with torch.cuda.device(0 if device=='cuda' else -1):
        macs, params = get_model_complexity_info(
            model, input_res=input_size, as_strings=False,
            print_per_layer_stat=False, verbose=False
        )

    assert type(macs) is int and type(params) is int, "macs/params not int"
    
    print(f"Trainable parameters:   {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"State dict size:        {param_size_MB:.2f} MB")
    print(f"FLOPs for forward pass: {2*macs / 1e9:.2f} GFLOPs")  # 1 MACs = 2 FLOPs -ish

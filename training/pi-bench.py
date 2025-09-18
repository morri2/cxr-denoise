# for the u-nets with base features 16, 32 and 64 run on random data and benchmark infrence without gradients (batch size 1)

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from unet_model import ResUNetSE

def benchmark_unet(features, in_channels=1, out_channels=1, img_size=512, device='cpu'):
    model = ResUNetSE(in_channels=in_channels, out_channels=out_channels, features=features, se_reduction=int(features/4)).to(device)
    model.eval()
    
    # Create random input tensor
    x = torch.randn((1, in_channels, img_size, img_size)).to(device)
    
    print("benchmarking UNet with features:", features)
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(x)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    print(f"Features: {features}, Avg Inference Time: {avg_time:.6f} seconds")
    return avg_time


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
for features in [16, 32, 64]:
    benchmark_unet(features, device=device)







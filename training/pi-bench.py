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
import os

def benchmark_unet(features, in_channels=1, out_channels=1, img_size=512, device='cpu', n_warmups=10, n_runs=100):
    model = ResUNetSE(in_channels=in_channels, out_channels=out_channels, features=features, se_reduction=int(features/4)).to(device)
    model.eval()
    
    # Create random input tensor
    x = torch.randn((1, in_channels, img_size, img_size)).to(device)
    
    print("Benchmarking UNet with features:", features)

    # test
    test_start_time = time.time()
    with torch.no_grad():
        _ = model(x)
    test_end_time = time.time()
    print(f"  First Infrence Time: {test_end_time - test_start_time:.6f} seconds. (no warm-up)")

    # Warm-up
    with torch.no_grad():
        for _ in range(n_warmups):
            _ = model(x)
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(x)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / n_runs
    print(f"  Avg Inference Time: {avg_time:.6f} seconds. (of {n_runs}, with {n_warmups} warm-ups)")
    return avg_time


N_WANRMUPS = 1
N_RUNS = 5

device = 'cpu'
print(f"Using device: {device}")

if torch.cuda.is_available():
    print("this benchmark is not made for the GPU, but try it anyway if you want")

for size in [512, 1024]:
    print(f"\n\nImage size: {size}x{size}\n")
    for features in [16, 32, 64]:
        benchmark_unet(features, img_size=size, device=device, n_warmups=N_WANRMUPS, n_runs=N_RUNS)
        time.sleep(5) # sleep to un-warmup









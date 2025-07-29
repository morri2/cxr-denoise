import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from collections.abc import Sized
from torchvision.transforms import GaussianBlur


class PreprocessClean(nn.Module):
    def __init__(self, clip_min=0.02, clip_max=1.0, scale_factor=0.8):
        super().__init__()
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.scale_factor = scale_factor

    def forward(self, x):
        """
        x: (B, 1, H, W) clean image tensor
        Returns: clipped and scaled tensor
        """
        x = x * self.scale_factor
        x = torch.clamp(x, self.clip_min, self.clip_max)
        return x
    


class PolyNoiseStd(nn.Module):
    def __init__(self, a, b, c):
        """
        Polynomial: std = a * I^2 + b * I + c
        I ∈ [0, 1] assumed
        """
        super().__init__()
        self.a = a
        self.b = b
        self.c = c

    def forward(self, x):
        """
        x: (B, 1, H, W) image tensor, values assumed in [0, 1]
        Returns: per-pixel noise std (B, 1, H, W)
        """
        std = self.a * x ** 2 + self.b * x + self.c
        return std
    
POLY_NOISE_STD_512 = PolyNoiseStd(0.0899 / 2, 0.0214 / 2, 0.0081 / 2)
POLY_NOISE_STD_1024 = PolyNoiseStd(0.0899, 0.0214, 0.0081)

class ApplyNoise(nn.Module):
    def __init__(self, noise_std_fn=POLY_NOISE_STD_512, blur_sigma=2.5):
        """
        Args:
            noise_std_fn: callable or nn.Module, maps (B,1,H,W) → (B,1,H,W) with per-pixel std
            blur_sigma: standard deviation for Gaussian blur
        """
        super().__init__()
        self.noise_std_fn = noise_std_fn

        # Compute kernel size from sigma: usually kernel_size = 6*sigma + 1
        kernel_size = int(6 * blur_sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.blur = GaussianBlur(kernel_size=kernel_size, sigma=blur_sigma)

    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) clean image tensor in [0, 1]
        Returns:
            Noisy image tensor (B, 1, H, W), clipped to [0, 1]
        """
        # Apply blur (expects channel-first format)
        x_blurred = self.blur(x)

        # Compute per-pixel std
        with torch.no_grad():
            std_map = self.noise_std_fn(x_blurred)

        # Add Gaussian noise
        noise = torch.randn_like(x_blurred) * std_map
        noisy = x_blurred + noise

        # Clip to valid image range
        return torch.clamp(noisy, 0.0, 1.0)
import torch.nn as nn
import torch
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

class MedianFilter(nn.Module):
    def __init__(self, ksize=3):
        super().__init__()
        self.ksize = ksize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N,C,H,W), float in [0,1] or any range
        """
        pad = self.ksize // 2
        # pad edges
        x_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')  # (N,C,H+ks,H+ks)
        
        # extract sliding patches
        patches = x_padded.unfold(2, self.ksize, 1).unfold(3, self.ksize, 1)  # (N,C,H,W,ks,ks)
        patches = patches.contiguous().view(*patches.shape[:4], -1)  # flatten ks*ks dims
        
        # median along the last dimension
        out = patches.median(dim=-1).values  # (N,C,H,W)
        return out

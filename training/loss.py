import torch
import torchvision.transforms as transforms
import torch.nn as nn

class SpatialLoss(nn.Module):
    """Spatial loss function using MSE and L1 loss."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss() 
        
    def forward(self, pred, target):
        spatial_loss = self.mse(pred, target) + 0.5 * self.l1(pred, target)
        return spatial_loss
        

class FrequencyLoss(nn.Module):
    """Loss in the frequency domain using FFT."""
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss() 

    def forward(self, pred, target):
        # Frequency component (FFT magnitude)
        pred_fft = torch.abs(torch.fft.fft2(pred))
        target_fft = torch.abs(torch.fft.fft2(target))
        freq_loss = self.l1(pred_fft, target_fft)
        return freq_loss
    
class CombinedLoss(nn.Module):
    """Combined loss function that incorporates both spatial and frequency losses."""
    def __init__(self, spatial_weight=1.0, freq_weight=0.3):
        super().__init__()
        self.spatial_loss = SpatialLoss()
        self.freq_loss = FrequencyLoss()

        self.spatial_weight = spatial_weight
        self.freq_weight = freq_weight
        
    def forward(self, pred, target):
        return self.spatial_weight * self.spatial_loss(pred, target) + self.freq_weight * self.freq_loss(pred, target)
import torch
import torch.nn as nn

class SpatialLoss(nn.Module):
    """Spatial loss function using weighted MSE and L1 loss."""
    def __init__(self, mse_weight=1.0, l1_weight=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)
        spatial_loss = self.mse_weight * mse_loss + self.l1_weight * l1_loss
        return spatial_loss
    

class FrequencyLoss(nn.Module):
    """Loss in the frequency domain using FFT."""
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        pred_fft = torch.abs(torch.fft.fft2(pred))
        target_fft = torch.abs(torch.fft.fft2(target))
        freq_loss = self.l1(pred_fft, target_fft)
        return freq_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function that incorporates:
    - MSE loss (spatial)
    - L1 loss (spatial)
    - Frequency loss
    """
    def __init__(self, mse_weight=1.0, l1_weight=0.5, freq_weight=0.3):
        super().__init__()
        self.spatial_loss = SpatialLoss(mse_weight=1.0, l1_weight=1.0)  # We'll scale externally
        self.freq_loss = FrequencyLoss()

        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.freq_weight = freq_weight

        # Store last computed losses
        self._last_mse = None
        self._last_l1 = None
        self._last_freq = None

    def forward(self, pred, target):
        # Compute individual spatial losses
        self._last_mse = self.spatial_loss.mse(pred, target)
        self._last_l1 = self.spatial_loss.l1(pred, target)
        self._last_freq = self.freq_loss(pred, target)

        total_loss = (
            self.mse_weight * self._last_mse +
            self.l1_weight * self._last_l1 +
            self.freq_weight * self._last_freq
        )
        return total_loss

    def get_last_losses(self):
        """Return the individual loss components from the last forward pass."""

        if self._last_mse is None or self._last_l1 is None or self._last_freq is None:
            return None
        else:
            return {
                'mse_loss': self._last_mse.item(),
                'l1_loss': self._last_l1.item(),
                'freq_loss': self._last_freq.item(),
            } 
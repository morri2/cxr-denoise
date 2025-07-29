import torch
from cxr_plt import *
from tqdm import tqdm
from loss import CombinedLoss, SpatialLoss, FrequencyLoss
import matplotlib.pyplot as plt
import os
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from torchmetrics.functional import structural_similarity_index_measure as ssim
import torch.nn.functional as F

def evaluate_metrics_on_dataloader(model, dataloader, preproc, noiser, device, max_batches=None):
    model.eval()
    ssim_sum = 0.0
    psnr_sum = 0.0
    l1_sum = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = batch[0].to(device)
            clean = preproc(batch)
            noisy = noiser(clean)
            output = model(noisy)

            B = clean.size(0)
            total_samples += B

            # L1
            l1_sum += F.l1_loss(output, clean, reduction='sum').item()

            # PSNR = 10 * log10(1 / MSE_per_pixel)
            mse = F.mse_loss(output, clean, reduction='sum').item()
            psnr_sum += 10 * torch.log10(torch.tensor(1.0) / (mse / (B * clean.numel() / B))).item()

            # SSIM (expects (N,C,H,W) and values in [0,1])
            ssim_res = ssim(output, clean, data_range=1.0)
            assert type(ssim_res) is torch.Tensor
            ssim_batch = ssim_res.item()
            ssim_sum += ssim_batch * B

            if max_batches and batch_idx + 1 >= max_batches:
                break

    model.train()
    return {
        "SSIM": ssim_sum / total_samples,
        "PSNR": psnr_sum / total_samples,
        "L1": l1_sum / (total_samples * clean.numel() / B)
    }


def save_cxr_triplet(clean, noisy, output, img_name, out_dir="saved_out"):
    os.makedirs(out_dir, exist_ok=True)

    # Ensure inputs are (C, 1, H, W)
    def preprocess(tensor):
        if tensor.dim() == 3:  # (1, H, W)
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 2:  # (H, W)
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        return tensor.detach().cpu().clamp(0, 1)

    clean = preprocess(clean)
    noisy = preprocess(noisy)
    output = preprocess(output)

    # Stack them row-wise: 3*C images total, grouped into 3 rows
    full_stack = torch.cat([clean, noisy, output], dim=0)  # Shape: (3*C, 1, H, W)

    # Arrange in C columns and 3 rows
    grid = make_grid(full_stack, nrow=clean.size(0), padding=5)

    img = to_pil_image(grid)
    img.save(os.path.join(out_dir, img_name))


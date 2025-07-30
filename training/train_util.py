import torch
from cxr_plt import *
from tqdm import tqdm
from loss import CombinedLoss, SpatialLoss, FrequencyLoss
import matplotlib.pyplot as plt
import os
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics import MeanAbsoluteError

from torchmetrics import MeanAbsoluteError
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio


def evaluate_metrics_on_dataloader(model, dataloader, preproc, noiser, device, max_batches=None):
    model.eval()

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    l1_metric = MeanAbsoluteError().to(device)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = batch[0].to(device)
            clean = preproc(batch)
            noisy = noiser(clean)
            output = model(noisy)
            output = output.clamp(0.0, 1.0)

            psnr_metric.update(output, clean)
            ssim_metric.update(output, clean)
            l1_metric.update(output, clean)

            if max_batches and batch_idx + 1 >= max_batches:
                break

    model.train()

    return {
        "ssim": ssim_metric.compute().item(),
        "psnr": psnr_metric.compute().item(),
        "l1": l1_metric.compute().item(),
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


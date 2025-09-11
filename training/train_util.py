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

from typing import Any


import piqa
import piq
import csv
#psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
#ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
#l1_metric = MeanAbsoluteError().to(device)
#gmsd_metric = piq.GMSDLoss().to(device)
#ms_gmsd_metric = piq.MultiScaleGMSDLoss().to(device)

psnr_metric = lambda x, y: piq.psnr(x,y)
ssim_metric = lambda x, y: piq.ssim(x,y)
ms_ssim_metric = lambda x, y: piq.multi_scale_ssim(x,y)

gmsd_metric = lambda x, y: piq.gmsd(x,y)
ms_gmsd_metric = lambda x, y: piq.multi_scale_gmsd(x,y)

# TODO LPIPS?

DEFAULT_METRICS = {
    "psnr": psnr_metric,
    "ssim": ssim_metric,
    "ms_ssim": ms_ssim_metric,
    "gmsd": gmsd_metric,
    "ms_gmsd": ms_gmsd_metric,
}

def evaluate_metrics_on_dataloader(model, dataloader, preproc, noiser, device, metrics=DEFAULT_METRICS, max_samples=None, 
                                   csv_path=None, csv_row_header: None | tuple[str, Any]=None):
    model.eval()
    
    values = {name: 0.0 for name in metrics.keys()}
    samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = batch[0].to(device)
            clean = preproc(batch)
            noisy = noiser(clean)
            output = model(noisy)
            output = output.clamp(0.0, 1.0)

            for name, metric in metrics.items():
                values[name] += metric(output, clean).item() * len(batch)
            samples += len(batch)
            
            if max_samples is not None and samples >= max_samples:
                break

    for name in values.keys():
        values[name] /= samples

    if csv_path is not None:
        with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                # add a first row if the file is empty
                if f.tell() == 0:
                    header = [csv_row_header[0]] if csv_row_header is not None else []
                    header += list(values.keys())
                    writer.writerow(header)

                row = [str(csv_row_header[1])] if csv_row_header is not None else []
                row += [values[k] for k in list(values.keys())]

                writer.writerow(row)
       
    return values




def save_cxr_triplet(clean, noisy, output, img_out_path, rescale=True):
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

    if rescale:
        all_min = min(clean.min().item(), noisy.min().item(), output.min().item())
        all_max = max(clean.max().item(), noisy.max().item(), output.max().item())
        clean = (clean - all_min) / (all_max - all_min)
        noisy = (noisy - all_min) / (all_max - all_min)
        output = (output - all_min) / (all_max - all_min)

    # Stack them row-wise: 3*C images total, grouped into 3 rows
    full_stack = torch.cat([noisy, clean, output], dim=0)  # Shape: (3*C, 1, H, W)

    # Arrange in C columns and 3 rows
    grid = make_grid(full_stack, nrow=clean.size(0), padding=5)

    img = to_pil_image(grid)
    img.save(img_out_path)


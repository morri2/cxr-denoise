import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from nih_dataset import NIH_Dataset
from torch.utils.data import DataLoader
from noise import PreprocessClean, ApplyNoise
import torch


def extract_seeded_patch(images, patch_size=64, index=0):
    """
    images: (B, 1, H, W)
    Returns: (B, 1, patch_size, patch_size)
    Extracts patches in a reproducible way by seeding with index.
    """
    B, _, H, W = images.shape
    if H < patch_size or W < patch_size:
        raise ValueError("Patch size is larger than image dimensions.")

    torch.manual_seed(index)  # Seed based on index
    top = torch.randint(0, H - patch_size + 1, (B,))
    left = torch.randint(0, W - patch_size + 1, (B,))

    patches = []
    for i in range(B):
        patch = images[i:i+1, :, top[i]:top[i]+patch_size, left[i]:left[i]+patch_size]
        patches.append(patch)
    return torch.cat(patches, dim=0)

def tensor_to_cv(img_tensor):
    # img_tensor: (1, H, W), range [0,1]
    img_np = img_tensor.squeeze().detach().cpu().numpy()
    img_np = (img_np * 255).clip(0, 255).astype('uint8')
    return img_np

def side_by_side(clean, noisy):
    # All are tensors: (1, H, W)
    imgs = [tensor_to_cv(img) for img in [clean, noisy]]
    return cv2.hconcat(imgs)

def interactive_patch_viewer(
    dataloader,
    preproc,
    noiser,
    device,
    patch_size=64,
    output_dir="selected_patches",
    display_scale=4
):
    os.makedirs(output_dir, exist_ok=True)

    dataset_batches = list(dataloader)
    total = len(dataset_batches)
    index = 0
    window_name = "Patch Viewer"

    print("Controls:")
    print("← / a : Previous | → / d : Next | s : Save clean | q : Quit")

    while 0 <= index < total:
        batch = dataset_batches[index][0].to(device)
        clean_full = preproc(batch)
        clean_patch = extract_seeded_patch(clean_full, patch_size, index)
        noisy_patch = noiser(clean_patch)

        clean_np = tensor_to_cv(clean_patch[0])
        noisy_np = tensor_to_cv(noisy_patch[0])
        combined = cv2.hconcat([clean_np, noisy_np])
        combined_up = cv2.resize(
            combined,
            (combined.shape[1] * display_scale, combined.shape[0] * display_scale),
            interpolation=cv2.INTER_NEAREST
        )

        cv2.imshow(window_name, combined_up)

        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_path = os.path.join(output_dir, f"clean_patch_{index:04d}.png")
            cv2.imwrite(save_path, clean_np)
            print(f"Saved clean patch: {save_path}")
        elif key == ord('d') or key == 83:
            index = (index + 1) % total
        elif key == ord('a') or key == 81:
            index = (index - 1 + total) % total

    cv2.destroyAllWindows()



# Dataset
DATA_PATH = "../data/NIH_data_512"

nih_train = NIH_Dataset(DATA_PATH, split="train")
nih_val = NIH_Dataset(DATA_PATH, split="val")
nih_test = NIH_Dataset(DATA_PATH, split="test")

#LIMIT_DATA = None # For Using Full Dataset
LIMIT_DATA = [500, 300, 100] # For Testing

if LIMIT_DATA:
    nih_train = torch.utils.data.Subset(nih_train, range(LIMIT_DATA[0])) if LIMIT_DATA[0] else nih_train
    nih_val = torch.utils.data.Subset(nih_val, range(LIMIT_DATA[1])) if LIMIT_DATA[1] else nih_val
    nih_test = torch.utils.data.Subset(nih_test, range(LIMIT_DATA[2])) if LIMIT_DATA[2] else nih_test

BATCH_SIZE = 1

train_dl = DataLoader(nih_train, batch_size=BATCH_SIZE, pin_memory=True)
val_dl = DataLoader(nih_val, batch_size=BATCH_SIZE)
test_dl = DataLoader(nih_test, batch_size=BATCH_SIZE)

print(f"train/val/test: {len(nih_train)}/{len(nih_val)}/{len(nih_test)}")

preproc: PreprocessClean = PreprocessClean() # apply to clean images for ground truth
noiser: ApplyNoise = ApplyNoise(blur_sigma=1.0) # apply to preproc to get noisy features

sample = next(iter(train_dl))
print("Sample batch shape:", sample[0].shape)



interactive_patch_viewer(

    dataloader=val_dl,
    preproc=preproc,
    noiser=noiser,
    device="cpu",
    patch_size=64,
    output_dir="saved_patch_comparisons"
)
# v1.0
# Based on torchxrayvision, used under the MIT-licence. 
import os
from skimage.io import imread
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import torch.nn.functional as F


class NIH_Dataset(Dataset):
    """
    Loads data from NIH ChestX-ray14 dataset with updated CSV format:
    - 'Image Index' column contains full image paths (relative to dataset_root)
    - 'Train', 'Val', and 'Test' columns indicate split membership

    Parameters:
        dataset_root: Path to the dataset directory (not used for splits now)
        split: 'train', 'val', or 'test'
        img_root_override: Override for image folder (unused if paths are absolute)
        csvpath: Path to the CSV metadata file
        views: List of image views to include, or ["*"] for all
        unique_patients: If True, only one image per patient
        out_array_type: "torch" or "np"
        out_min, out_max: Normalized output image intensity range
        out_size: size of the image, if !=1024, image will be resized with interpolation
        img_max_val: Maximum intensity value in the raw image
        no_lbls: If True, only the image is returned
        preload_to_ram: Loads the full dataset to ram, requires >32gb ram
        to_device: .to(device) on the images before preprocessing
    """

    def __init__(self,
                 dataset_root,
                 split="train",
                 csvpath=None,
                 views=["PA", "AP"],
                 unique_patients=False,
                 out_array_type="torch",
                 out_min=0.0,
                 out_max=1.0,
                 out_size=512, # will bi-lerp to resize - only for torch
                 img_max_val=255.0,
                 no_lbls=False,
                 preload_to_ram=False, # Warning! Not advisable for machines with <32gb RAM.
                 to_device=None,
                 ):

        super(NIH_Dataset, self).__init__()

        assert split in {"train", "val", "test"}, "Invalid split specified."
        self.split = split
        self.img_max_val = img_max_val
        self.out_min = out_min
        self.out_max = out_max
        self.out_size = out_size
        self.out_array_type = out_array_type
        self.no_lbls = no_lbls
        self.dataset_root = dataset_root
        self.preload_to_ram = preload_to_ram
        self.to_device = to_device

        self.pathologies = [
            "Atelectasis", "Consolidation", "Infiltration",
            "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
            "Effusion", "Pneumonia", "Pleural_Thickening",
            "Cardiomegaly", "Nodule", "Mass", "Hernia"
        ]

        # Load CSV
        if csvpath is None:
            csvpath = os.path.join(dataset_root, "EnhancedDataEntry.csv")

        self.csv = pd.read_csv(csvpath)

        # Standardize columns
        self.csv["view"] = self.csv["View Position"].fillna("UNKNOWN")
        self.csv["patientid"] = self.csv["Patient ID"].astype(str)
        self.csv["age_years"] = self.csv["Patient Age"] * 1.0
        self.csv["sex_male"] = self.csv["Patient Gender"] == 'M'
        self.csv["sex_female"] = self.csv["Patient Gender"] == 'F'

        # Filter by split
        split_col = {"train": "Train", "val": "Val", "test": "Test"}[split]
        self.csv = self.csv[self.csv[split_col] == "Yes"]

        # Filter by view
        self.views = views if isinstance(views, list) else [views]
        if "*" not in self.views:
            self.csv = self.csv[self.csv["view"].isin(self.views)]

        # Unique patient filter
        if unique_patients:
            self.csv = self.csv.groupby("Patient ID").first().reset_index()

        # Store image paths
        self.img_paths = self.csv["Image Index"].apply(lambda p: p.replace("\\", "/")).tolist()

        # Compute labels
        self.labels = np.stack([
            self.csv["Finding Labels"].str.contains(p).fillna(False).values.astype(np.float32)
            for p in self.pathologies
        ], axis=1)
        print("NIH CXR Dataset ({}x{}), split='{}' loaded with {} samples".format(out_size, out_size, split, len(self.labels)))

        self.ram_imgs = None
        if preload_to_ram:
            print("  Loading to RAM... ")
            self.ram_imgs = np.empty((len(self.labels), 1, self.out_size, self.out_size), dtype=np.uint8)
            total_bytes = 0
            
            for idx in range(len(self.labels)):
                img_path = os.path.join(self.dataset_root, self.img_paths[idx])
                img = imread(img_path).astype(np.uint8)

                if img.ndim > 2: # for the rare rgba images in the dataset
                    img = np.mean(img, axis=2).astype(np.uint8)
                
                assert self.out_size != img.shape[-1], "Can't preload to RAM if self.out_size != img.shape[-1]!"
                # TODO: fix - low prio

                self.ram_imgs[idx] = img
                total_bytes += img.nbytes
            
            print(f"  Done! Using {total_bytes / (1024**3):.1f} GB of RAM")


    def string(self):
        return f"{self.__class__.__name__} num_samples={len(self)} views={self.views} unique_patients={self.split}"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.ram_imgs is None:
            # Load image from disk
            img_path = os.path.join(self.dataset_root, self.img_paths[idx])
            img = imread(img_path).astype(np.float32)

            if img.ndim > 2: # for the rare rgba images in the dataset
                img = np.mean(img, axis=2)

            img = img[None, :, :]  # add channel dim

        else: 
            # Use image from ram
            img = self.ram_imgs[idx].astype(np.float32) # imgs are saved as uin8s to save memory
        
        lbl = self.labels[idx]

        img = torch.from_numpy(img)
        lbl = torch.from_numpy(lbl)

        if self.to_device:
            img = img.to(device=self.to_device)
            lbl = img.to(device=self.to_device)

        # Normalize
        img = (img / self.img_max_val) * (self.out_max - self.out_min) + self.out_min
        
        if not self.preload_to_ram:
            if self.out_size != img.shape[-1]: # resize only if not the assumed size
                img = F.interpolate(img, (self.out_size, self.out_size),mode='bilinear')

        if self.out_array_type == "np":
            img = img.numpy()
            lbl = lbl.numpy()
        
        return img if self.no_lbls else (img, lbl)
        
    
def extract_random_patch(image, patch_size=64):
    """
    Extract a random patch from a single image tensor of shape (1, H, W).
    Returns a patch of shape (1, patch_size, patch_size).
    """
    _, H, W = image.shape
    ph, pw = patch_size, patch_size

    if H < ph or W < pw:
        raise ValueError("Patch size is larger than image dimensions.")

    top = torch.randint(0, H - ph + 1, size=(1,)).item()
    left = torch.randint(0, W - pw + 1, size=(1,)).item()

    patch = image[:, top:top + ph, left:left + pw]
    return patch


class RandomPatchDataset(Dataset):
    def __init__(self, dataset, patch_size):
        """
        Args:
            dataset: A dataset that returns (image, label) or just image tensors
            patch_size: Size of the square patch to extract
        """
        self.dataset = dataset
        self.patch_size = patch_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = data[0] if isinstance(data, (tuple, list)) else data
        return extract_random_patch(img, patch_size=self.patch_size)

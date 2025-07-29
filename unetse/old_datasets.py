import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

class FastNIHDataset(Dataset):
    def __init__(self, root_dir, split='train', cache_mode='last'):
        """
        split: one of 'train', 'val', or 'test'
        cache_mode: one of 'none', 'last', or 'all'
        """
        assert cache_mode in ('none', 'last', 'all'), "Invalid cache_mode"
        self.root_dir = Path(root_dir)
        self.split = split
        self.cxrs_dir = self.root_dir / split / 'cxrs'
        self.lbls_dir = self.root_dir / split / 'lbls'

        self.cxr_files = sorted(self.cxrs_dir.glob('*.npz'))
        self.lbl_files = sorted(self.lbls_dir.glob('*.npz'))
        assert len(self.cxr_files) == len(self.lbl_files), "Mismatch between image and label files"

        self.samples_per_file = 1000
        self.total_files = len(self.cxr_files)

        # Determine dataset length
        last_cxr = np.load(self.cxr_files[-1])
        self.last_file_len = last_cxr['arr_0'].shape[0]
        last_cxr.close()
        self.length = (self.total_files - 1) * self.samples_per_file + self.last_file_len

        self.cache_mode = cache_mode
        self.cache_stats = {'hits': 0, 'misses': 0}

        # Cache containers
        self._all_cached = None
        self._last_file_idx = None
        self._cached_cxrs = None
        self._cached_lbls = None

        if self.cache_mode == 'all':
            print(f"📦 Caching all {len(self.cxr_files)} files into memory...")
            self._all_cached = []
            for cxr_file, lbl_file in zip(self.cxr_files, self.lbl_files):
                cxrs = np.load(cxr_file)['arr_0']
                lbls = np.load(lbl_file)['arr_0']
                self._all_cached.append((cxrs, lbls))
            print("✅ All files cached.")

    def __len__(self):
        return self.length

    def _load_batch(self, file_idx):
        if self.cache_mode == 'all':
            self.cache_stats['hits'] += 1
            return self._all_cached[file_idx]

        elif self.cache_mode == 'last':
            if file_idx == self._last_file_idx:
                self.cache_stats['hits'] += 1
                return self._cached_cxrs, self._cached_lbls
            else:
                self.cache_stats['misses'] += 1
                cxrs = np.load(self.cxr_files[file_idx])['arr_0']
                lbls = np.load(self.lbl_files[file_idx])['arr_0']
                self._cached_cxrs = cxrs
                self._cached_lbls = lbls
                self._last_file_idx = file_idx
                return cxrs, lbls

        else:  # 'none'
            self.cache_stats['misses'] += 1
            cxrs = np.load(self.cxr_files[file_idx])['arr_0']
            lbls = np.load(self.lbl_files[file_idx])['arr_0']
            return cxrs, lbls

    def __getitem__(self, idx):
        file_idx = idx // self.samples_per_file
        inner_idx = idx % self.samples_per_file

        cxrs, lbls = self._load_batch(file_idx)
        cxr = cxrs[inner_idx]
        lbl = lbls[inner_idx]

        cxr_tensor = torch.from_numpy(cxr).unsqueeze(0).float() / 255.0
        lbl_tensor = torch.from_numpy(lbl).float()

        return cxr_tensor, lbl_tensor
    




class DenoiseCXRDataset(Dataset):
    def __init__(self, base_dataset: Dataset):
        """
        Args:
            base_dataset: Dataset that returns (image, label) tuples
            noiser: nn.Module that adds noise to an image tensor
        """
        self.base_dataset = base_dataset
        kernel_size = 5
        sigma = 2.5
        self.blur  = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

        cxr_size = base_dataset[0][0].shape[-1]

        if cxr_size == 1024:
            # 1024 x 1024 std
            self.std_from_val = lambda T: 0.0899 * T**2 + 0.0214 * T + 0.0081
        elif cxr_size == 512:
            # 512 x 512 std
            self.std_from_val = lambda T: (0.0899 * T**2 + 0.0214 * T + 0.0081) / 2 # derived from 1024 std / 2 (since it is mean of 4 pixels)
        else: 
            raise ValueError(f"Unsupported CXR size: {cxr_size}. Expected 512 or 1024.")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        clean_img, _ = self.base_dataset[idx]  # label not needed

        # preprocessing
        clean_img: torch.Tensor = clean_img * 0.8
        clean_img = clean_img.clamp(0.02, 1.0)

        noisy_img = self.blur(clean_img)  # Apply Gaussian blur
        noisy_img = torch.normal(noisy_img,)
        noisy_img = noisy_img.clamp(0.0, 1.0)

        return noisy_img, clean_img  # (input, target)
    

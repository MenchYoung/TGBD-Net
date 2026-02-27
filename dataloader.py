import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import monai.transforms as transforms
from monai.data import ThreadDataLoader
from monai.transforms import LoadImage, EnsureChannelFirst
import os
import SimpleITK as sitk
import time


# ============================================================
# 1. HybridLoader:
# ============================================================
class HybridLoader:
    """
    Unified loader for mixed storage formats.
    1. .pt -> torch.load -> (C, D, H, W)
    2. .nii/.nii.gz -> MONAI LoadImage -> (H, W, D) -> Permute -> (C, D, H, W)
    """
    def __init__(self):

        # MONAI loader:
        # - EnsureChannelFirst(channel_dim="no_channel") forces an explicit channel dimension.
        # - image_only=True returns only the image tensor (no metadata dict).
        self.nii_loader = transforms.Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(channel_dim="no_channel") 
        ])

    def __call__(self, path):
        path = str(path).strip()
        
        # --- .pt tensors ---
        if path.endswith('.pt'):
            data = torch.load(path)
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            data = data.float()
            # (C, D, H, W)
            if data.ndim == 3: 
                data = data.unsqueeze(0)
            return data

        # ---  NIfTI volumes (.nii / .nii.gz) ---
        elif path.endswith('.nii') or path.endswith('.nii.gz'):

            data = self.nii_loader(path) 
            if data.ndim == 4:
                data = data.permute(0, 3, 1, 2)
            
            return data.float()

        else:
            raise ValueError(f"Unsupported file format: {path}")

# ============================================================
# 2. Dataset (with PFS fields)
# ============================================================
class EGFRBboxDataset(Dataset):
    def __init__(self, csv_path, mode, image_size):
        self.mode = mode
        self.image_size = image_size
        
        full_df = pd.read_csv(csv_path)
        
        # 1. Split 
        df_mode_subset = full_df[full_df['split'] == self.mode].reset_index(drop=True)
        
        self.df_filtered = df_mode_subset
        
        # Column names
        self.global_path_col = 'xxxxxx'
        self.local_path_col = 'xxxxxx'
        
        print(f"--- [{self.mode.upper()}] Hybrid Dataset ---")
        print(f"  - Total samples: {len(self.df_filtered)}")
        
        # Diagnostics: number of samples with valid PFS annotations
        n_pfs = (self.df_filtered['xxxxxx'] > 0).sum()
        print(f"  - Samples with PFS: {n_pfs} ")

        # --- Transforms ---
        self.spatial_aug = transforms.Compose([
            transforms.RandFlipd(keys=["image_global", "image_local"], prob=0.5, spatial_axis=[0, 1, 2], allow_missing_keys=True),
        ])
        
        self.post_process = transforms.Compose([
            transforms.Resized(keys=["xxxxxx", "xxxxxx"], spatial_size=self.image_size, mode="trilinear", allow_missing_keys=True),
            transforms.ScaleIntensityRanged(keys=["xxxxxx", "xxxxxx"], a_min=-1200.0, a_max=400.0, b_min=0.0, b_max=1.0, clip=True, allow_missing_keys=True),
        ])
        
        if self.mode == 'train':
            self.random_cropper = transforms.RandSpatialCrop(roi_size=self.image_size, random_size=False)
        else:
            self.random_cropper = transforms.CenterSpatialCrop(roi_size=self.image_size)

        self.loader = HybridLoader()

    def __len__(self):
        return len(self.df_filtered)

    def __getitem__(self, idx,swap_count =0):

        MAX_RETRIES = 10   # retries for the same sample (e.g., transient I/O issues)
        MAX_SWAPS = 5      # maximum fallback samples before raising a fatal error
        

        if swap_count >= MAX_SWAPS:
            raise RuntimeError(
                f"❌ [Fatal Error] Failed after swapping {swap_count} samples. "
                f"Please verify dataset paths and storage connectivity."
            )


        for attempt in range(MAX_RETRIES):
            try:
                row = self.df_filtered.loc[idx]
                
                # --- 1. Load Image (auto-detect .pt/.nii) ---
                global_path = str(row[self.global_path_col])
                image_global = self.loader(global_path)
                
                has_bbox_flag = int(row['xxxxxx'])
                local_path = str(row[self.local_path_col])
                
                if has_bbox_flag == 1 and not pd.isna(local_path) and os.path.exists(local_path):
                    image_local = self.loader(local_path)
                else:
                    image_local = self.random_cropper(image_global)
                    has_bbox_flag = 0

                # --- 2. Apply spatial augmentation and post-processing ---
                data_dict = {"image_global": image_global, "image_local": image_local}
                if self.mode == 'train':
                    data_dict = self.spatial_aug(data_dict)
                transformed_data = self.post_process(data_dict)
                
                img_g = transformed_data['xxxxx']
                img_l = transformed_data['xxxxx']


                # Replicate to 3 channels if single-channel
                if img_g.shape[0] == 1: img_g = img_g.repeat(3, 1, 1, 1)
                if img_l.shape[0] == 1: img_l = img_l.repeat(3, 1, 1, 1)
                
                # --- 3. Read PFS labels ---
                # if absent, set to -1
                pfs_time = float(row.get('xxxxxx', -1.0))
                pfs_event = float(row.get('xxxxxx', -1.0))
                
                if pd.isna(pfs_time): pfs_time = -1.0
                if pd.isna(pfs_event): pfs_event = -1.0
                
                # read egfr labels
                label_egfr = int(row['xxxxxx'])

                return {
                    "image_global": img_g,
                    "image_local": img_l,
                    "label_egfr": torch.tensor(label_egfr, dtype=torch.long),
                    "pfs_time": torch.tensor(pfs_time, dtype=torch.float),   
                    "pfs_event": torch.tensor(pfs_event, dtype=torch.float), 
                    "has_bbox": torch.tensor(has_bbox_flag, dtype=torch.long),
                    "id": str(row.get('xxxxxx', idx))
                }
            
            
            except (FileNotFoundError, OSError, RuntimeError, Exception) as e:
                wait_time = (attempt + 1) * 0.2
                if attempt % 2 == 0:
                    print(f"⚠️ [Data Error] ID:{row.get('xxxxxx', idx)} | Try {attempt+1}/{MAX_RETRIES} Error: {e}")
                
                time.sleep(wait_time)
                
                if attempt == MAX_RETRIES - 1:
                    print(f"❌ [Give Up] ID {idx} cannot be loaded. Swapping...")
                    
                    new_idx = idx
                    while new_idx == idx and len(self.df_filtered) > 1:
                        new_idx = np.random.randint(0, len(self.df_filtered))
                    
                    print(f"🔄 [Swapping] new ID {new_idx} Number of switches: {swap_count + 1}/{MAX_SWAPS})")
                    
                    return self.__getitem__(new_idx, swap_count=swap_count + 1)

# ============================================================
# 3. Collate Fn
# ============================================================
def bbox_collate_fn(batch):
    return {
        'image_global': torch.stack([item['xxxxxx'] for item in batch]),
        'image_local': torch.stack([item['xxxxxx'] for item in batch]),
        'label_egfr': torch.stack([item['xxxxxx'] for item in batch]),
        'pfs_time': torch.stack([item['xxxxxx'] for item in batch]),   
        'pfs_event': torch.stack([item['xxxxxx'] for item in batch]), 
        'has_bbox': torch.stack([item['xxxxxx'] for item in batch]),
        'id': [item['xxxxxx'] for item in batch]
    }

# ============================================================
# 4. DataLoader builder function, with WeightedRandomSampler
# ============================================================
def get_bbox_dataloaders(csv_path, image_size, batch_size=4, num_workers=4):
    dls = {}
    
    # Read CSV once for sampling weights (train split only)
    full_df = pd.read_csv(csv_path)
    
    for mode in ['train', 'val', 'test']:
        dataset = EGFRBboxDataset(csv_path=csv_path, mode=mode, image_size=image_size)
        
        sampler = None
        shuffle = (mode == 'train')
        
        # Enable WeightedRandomSampler for training
        if mode == 'train':
            # 1. Extract all PFS infomation in this split
            train_df = full_df[full_df['split'] == 'train'].reset_index(drop=True)
            
            # 2. Define weights
            #    - samples with PFS: higher weight (more likely to be sampled)
            #    - samples without PFS: lower weight
            has_pfs = (train_df['xxxxxx'] > 0).astype(float)
            has_pfs_mask = (train_df['pfs_time'] > 0)
            n_pfs = has_pfs_mask.sum()
            n_egfr_only = len(train_df) - n_pfs
            
            weights = np.where(has_pfs, 1.0, 0.2) 
            
            weights = torch.DoubleTensor(weights)
            
            # 3. Define virtual length
            #    - maintaining sufficient PFS density in batches
            #    - achieving broad coverage of EGFR-only samples across an epoch (in expectation)
            virtual_length = int(n_egfr_only * 1.0 + n_pfs * 5.0)
            
            print(f"  -> Extending train set length from {len(weights)} to {virtual_length} )")

            # 4. Create WeightedRandomSampler
            sampler = WeightedRandomSampler(weights, num_samples=virtual_length, replacement=True)
            shuffle = False
            
            print(f"  -> [Sampler] WeightedRandomSampler with PFS weight {1.0} and EGFR-only weight {0.2}")

        dls[mode] = ThreadDataLoader(
            dataset, batch_size=batch_size, 
            sampler=sampler, shuffle=shuffle,
            num_workers=num_workers, collate_fn=bbox_collate_fn, pin_memory=True
        )
    return dls


def get_kmeans_dataloaders(csv_path, image_size, batch_size=4, num_workers=4):
    """
    DataLoader used specifically for K-means initialization.

    Characteristics:
      - train split only
      - no sampler
      - deterministic order (shuffle=False)
    """

    
    # Only perform on train split
    dataset = EGFRBboxDataset(csv_path=csv_path, mode='train', image_size=image_size)
    
    loader = ThreadDataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=None, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=bbox_collate_fn, 
        pin_memory=True
    )
    

    return loader 

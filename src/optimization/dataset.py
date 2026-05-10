import os
import h5py
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class TEFilmDataset(Dataset):
    """
    PyTorch Dataset for TE Film Optimization.
    Reads metadata from CSV for scalar inputs and targets,
    and loads the 3D thermal conductivity mask from HDF5 on-the-fly.
    """
    def __init__(
        self,
        metadata_csv,
        root_dir,
        transform=None,
        normalize_target=False,
        return_weight=False,
        top_quantile=None,
        top_weight=1.0,
    ):
        """
        Args:
            metadata_csv (string): Path to the metadata.csv file.
            root_dir (string): Directory where the `fields/` folder is located.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.normalize_target = normalize_target
        self.return_weight = return_weight
        # Only keep successful simulations that have a delta_T_parallel
        df = pd.read_csv(metadata_csv, low_memory=False)
        
        # ENHANCEMENT: Force scalar columns to numeric to handle any lingering corrupt strings
        self.scalar_cols = ['thickness_h', 'k_low', 'k_high', 'T_hot', 'T_air']
        for col in self.scalar_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Also ensure the target column is numeric
        self.target_col = 'delta_T_parallel'
        df[self.target_col] = pd.to_numeric(df[self.target_col], errors='coerce')

        # Drop any rows that failed to parse (NaNs in our required columns)
        df = df.dropna(subset=self.scalar_cols + [self.target_col, 'qc_pass'])
        
        self.data_frame = df[df['qc_pass'] == True].reset_index(drop=True)
        
        # Calculate statistics for normalization (Z-score)
        self.scalar_mean = self.data_frame[self.scalar_cols].mean().values
        self.scalar_std = self.data_frame[self.scalar_cols].std().values
        # Prevent division by zero if std is very small
        self.scalar_std = np.where(self.scalar_std < 1e-6, 1e-6, self.scalar_std)
        
        # Also store target stats in case we want to normalize targets later
        self.target_mean = self.data_frame[self.target_col].mean()
        self.target_std = self.data_frame[self.target_col].std()
        if pd.isna(self.target_std) or self.target_std < 1e-6:
            self.target_std = 1e-6

        self.sample_weights = np.ones(len(self.data_frame), dtype=np.float32)
        if top_quantile is not None and top_weight > 1.0:
            if not 0.0 < top_quantile < 1.0:
                raise ValueError("top_quantile must be between 0 and 1.")
            cutoff = self.data_frame[self.target_col].quantile(top_quantile)
            is_top = self.data_frame[self.target_col] >= cutoff
            self.sample_weights[is_top.to_numpy()] = np.float32(top_weight)
            self.top_weight_cutoff = float(cutoff)
        else:
            self.top_weight_cutoff = None

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_frame.iloc[idx]
        
        # 1. Load Scalars (Physics Params)
        scalars = row[self.scalar_cols].values.astype(np.float32)
        # Normalize scalars
        scalars_norm = (scalars - self.scalar_mean) / self.scalar_std
        
        # 2. Load 3D Voxel Mask
        h5_rel_path = row['field_file']
        # If absolute path is saved in CSV, we might need to extract the filename
        h5_filename = os.path.basename(h5_rel_path)
        h5_path = os.path.join(self.root_dir, 'fields', h5_filename)
        
        with h5py.File(h5_path, 'r') as f:
            k_map = f['fields/kappa'][:]  # Correct key is 'kappa', not 'thermal_conductivity'
            
        # Convert k_map to a binary mask (0 for k_low, 1 for k_high)
        # Using the midpoint as threshold
        threshold = (row['k_low'] + row['k_high']) / 2.0
        mask_3d = (k_map > threshold).astype(np.float32)
        
        # Add channel dimension for PyTorch 3D CNN: (C, D, H, W) -> (1, nx, ny, nz)
        mask_tensor = torch.from_numpy(mask_3d).unsqueeze(0).float()  # Ensure float32
        scalar_tensor = torch.from_numpy(scalars_norm).float()  # Ensure float32
        
        # 3. Target
        target_value = row[self.target_col]
        if self.normalize_target:
            target_value = (target_value - self.target_mean) / self.target_std
        target = np.array([target_value], dtype=np.float32)
        target_tensor = torch.from_numpy(target)

        if self.return_weight:
            weight = torch.tensor([self.sample_weights[idx]], dtype=torch.float32)
            return mask_tensor, scalar_tensor, target_tensor, weight

        return mask_tensor, scalar_tensor, target_tensor

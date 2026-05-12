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
        include_boundary_channel=False,
        scalar_cols=None,
        check_field_files=True,
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
        self.include_boundary_channel = include_boundary_channel
        self.input_channels = 2 if include_boundary_channel else 1
        self.check_field_files = check_field_files
        # Only keep successful simulations that have a delta_T_parallel
        df = pd.read_csv(metadata_csv, low_memory=False)
        
        # ENHANCEMENT: Force scalar columns to numeric to handle any lingering corrupt strings
        base_scalar_cols = ['thickness_h', 'k_low', 'k_high', 'T_hot', 'T_air']
        expanded_scalar_cols = [
            'k_ratio', 'h_c', 'h_c_side',
            'convection_regime_code', 'hot_boundary_type_code',
            'T_hot_min', 'T_hot_max', 'T_hot_amplitude',
            'gradient_direction_code', 'hotspot_x', 'hotspot_y', 'hotspot_sigma',
            'curvature_level', 'arc_angle', 'bend_axis_code',
            'bend_radius', 'arc_length', 'projected_length',
            'projected_Lx', 'projected_Ly',
        ]
        if scalar_cols is not None:
            self.scalar_cols = list(scalar_cols)
        else:
            self.scalar_cols = base_scalar_cols + [
                col for col in expanded_scalar_cols
                if col in df.columns and not df[col].isna().all()
            ]
        for col in self.scalar_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Also ensure the target column is numeric
        self.target_col = 'delta_T_parallel'
        df[self.target_col] = pd.to_numeric(df[self.target_col], errors='coerce')

        # Drop any rows that failed to parse (NaNs in our required columns)
        df = df.dropna(subset=self.scalar_cols + [self.target_col, 'qc_pass'])
        
        self.data_frame = df[df['qc_pass'] == True].reset_index(drop=True)
        if self.check_field_files:
            field_exists = self.data_frame['field_file'].map(lambda value: os.path.exists(self._field_path(value)))
            missing_count = int((~field_exists).sum())
            if missing_count:
                missing_examples = [
                    self._field_filename(value)
                    for value in self.data_frame.loc[~field_exists, 'field_file'].head(5)
                ]
                print(
                    f"Warning: dropping {missing_count} rows with missing HDF5 field files. "
                    f"Examples: {missing_examples}"
                )
                self.data_frame = self.data_frame[field_exists].reset_index(drop=True)
        
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
        self.hot_boundary_mean = float(self.data_frame['T_hot'].mean())
        self.hot_boundary_std = float(self.data_frame['T_hot'].std())
        if pd.isna(self.hot_boundary_std) or self.hot_boundary_std < 1e-6:
            self.hot_boundary_std = 1e-6

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

    def _field_filename(self, field_file):
        # Metadata may be generated on Windows and trained on Linux; normalize both
        # separator styles before taking the filename.
        return os.path.basename(str(field_file).replace('\\', '/'))

    def _field_path(self, field_file):
        return os.path.join(self.root_dir, 'fields', self._field_filename(field_file))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_frame.iloc[idx]
        
        # 1. Load Scalars (Physics Params)
        scalars = row[self.scalar_cols].values.astype(np.float32)
        # Normalize scalars
        scalars_norm = (scalars - self.scalar_mean) / self.scalar_std
        
        # 2. Load 3D Voxel Mask
        h5_path = self._field_path(row['field_file'])
        
        with h5py.File(h5_path, 'r') as f:
            k_map = f['fields/kappa'][:]  # Correct key is 'kappa', not 'thermal_conductivity'
            if self.include_boundary_channel and 'fields/hot_boundary_temperature' in f:
                hot_boundary = f['fields/hot_boundary_temperature'][:]
            else:
                hot_boundary = None
            
        # Convert k_map to a binary mask (0 for k_low, 1 for k_high)
        # Using the midpoint as threshold
        threshold = (row['k_low'] + row['k_high']) / 2.0
        mask_3d = (k_map > threshold).astype(np.float32)
        channels = [mask_3d]
        if self.include_boundary_channel:
            if hot_boundary is None:
                hot_boundary = np.full(mask_3d.shape[:2], float(row['T_hot']), dtype=np.float32)
            hot_norm = (hot_boundary.astype(np.float32) - self.hot_boundary_mean) / self.hot_boundary_std
            hot_channel = np.repeat(hot_norm[:, :, np.newaxis], mask_3d.shape[2], axis=2)
            channels.append(hot_channel.astype(np.float32))
        
        # Add channel dimension for PyTorch 3D CNN: (C, D, H, W) -> (1, nx, ny, nz)
        mask_tensor = torch.from_numpy(np.stack(channels, axis=0)).float()  # Ensure float32
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

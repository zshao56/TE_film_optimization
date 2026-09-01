import torch
import torch.nn as nn

class ConvBlock3D(nn.Module):
    """Two 3x3x3 convolutions with GroupNorm, optionally halving the resolution.

    GroupNorm rather than BatchNorm: v3 batches are grouped by grid shape and
    sized by voxel count (32 for 200x200x20 up to 512 for 50x50x20), so batch
    statistics differ systematically between batches. BatchNorm's running stats
    ended up dominated by whichever shapes happened to close an epoch, which
    made eval-mode validation loss swing between 0.30 and 0.71 while training
    loss fell smoothly. GroupNorm normalizes within each sample, so train and
    eval behave identically regardless of batch size or grid shape.
    """
    def __init__(self, in_channels, out_channels, pool=True, num_groups=8):
        super(ConvBlock3D, self).__init__()
        groups = min(num_groups, out_channels)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool3d(2) if pool else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x)

class ThermoNetFusion(nn.Module):
    def __init__(self, scalar_dim=5, input_channels=1):
        super(ThermoNetFusion, self).__init__()
        
        # 1. 3D CNN Branch. v3 grids range from 50x50x20 to 200x200x20, so the
        # adaptive pool below is what makes a single model handle every shape.
        #
        # It pools to 4x4x2 rather than 1x1x1. Global average pooling collapsed
        # the field to a per-channel mean, which discards where the conductive
        # material sits: two structures with the same mean but different heat
        # paths became identical features. delta_T depends on that layout, and
        # the symptom was per-band R2 going negative (-12.2 in the 10-15K band)
        # while overall R2 stayed high at 0.909 from coarse magnitude alone.
        self.cnn_branch = nn.Sequential(
            ConvBlock3D(input_channels, 32, pool=True),    # (B, 32, nx/2, ny/2, nz/2)
            ConvBlock3D(32, 64, pool=True),                # (B, 64, nx/4, ny/4, nz/4)
            ConvBlock3D(64, 128, pool=False),              # (B, 128, nx/4, ny/4, nz/4)
            nn.AdaptiveAvgPool3d((4, 4, 2)),               # (B, 128, 4, 4, 2)
            nn.Flatten(),                                  # (B, 4096)
            nn.Linear(128 * 4 * 4 * 2, 256),
            nn.ReLU(inplace=True),
        )
        
        # 2. Scalar MLP Branch (Processing Physical Params)
        # Input: (B, 5) -> [h, k_low, k_high, T_hot, T_air]
        self.mlp_branch = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True)
        )
        
        # 3. Fusion Block
        # Combines CNN feature vector (256) + MLP feature vector (32) = 288
        self.fusion_head = nn.Sequential(
            nn.Linear(256 + 32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)  # Predicts delta_T_parallel
        )

    def forward(self, mask_3d, scalars):
        # Extract geometry features
        geo_features = self.cnn_branch(mask_3d)
        
        # Extract physics features
        phys_features = self.mlp_branch(scalars)
        
        # Concatenate
        combined = torch.cat((geo_features, phys_features), dim=1)
        
        # Regress
        out = self.fusion_head(combined)
        return out

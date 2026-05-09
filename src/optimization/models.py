import torch
import torch.nn as nn

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool3d(2) if pool else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x)

class ThermoNetFusion(nn.Module):
    def __init__(self, scalar_dim=5):
        super(ThermoNetFusion, self).__init__()
        
        # 1. 3D CNN Branch (Processing 50x50x20 Voxel Mask)
        # Input: (B, 1, 50, 50, 20)
        self.cnn_branch = nn.Sequential(
            ConvBlock3D(1, 16, pool=True),    # Output: (B, 16, 25, 25, 10)
            ConvBlock3D(16, 32, pool=True),   # Output: (B, 32, 12, 12, 5)
            ConvBlock3D(32, 64, pool=False),  # Output: (B, 64, 12, 12, 5)
            nn.AdaptiveAvgPool3d(1),          # Output: (B, 64, 1, 1, 1) - Global Average Pooling
            nn.Flatten()                      # Output: (B, 64)
        )
        
        # 2. Scalar MLP Branch (Processing Physical Params)
        # Input: (B, 5) -> [h, k_low, k_high, T_hot, T_air]
        self.mlp_branch = nn.Sequential(
            nn.Linear(scalar_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True)
        )
        
        # 3. Fusion Block
        # Combines CNN feature vector (64) + MLP feature vector (32) = 96
        self.fusion_head = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)  # Predicts delta_T_parallel
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

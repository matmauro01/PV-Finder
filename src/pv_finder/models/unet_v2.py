"""UNet_1000_v2 — sidelobe-free K2H architecture.

Replaces ConvTranspose1d upsampling with nearest-neighbor interpolation + Conv1d,
reduces bottleneck from 8x to 4x, and uses additive skip connections with
residual encoder blocks. See docs/models/vertex_finding.md for rationale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pv_finder.models.autoencoder_models import ConvBNrelu, ResConvBNrelu


class InterpUp(nn.Module):
    """Upsample 2x via nearest-neighbor interpolation, then Conv1d + BN + ReLU."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, p: float = 0
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet_1000_v2(nn.Module):
    """K2H UNet with interpolation upsampling, 2-level pooling, additive skips.

    Architecture (default n=64, n_features=1):
        Encoder:  ConvBNrelu(1→64, k=15) → ResConv(64, k=5) → pool
                  → ResConv(64, k=5) → pool → ResConv(64, k=5)
        Decoder:  InterpUp(64→64) + skip → ResConv(64, k=5)
                  InterpUp(64→64) + skip → ResConv(64, k=5)
        Head:     Conv(64→64, k=5) → Conv(64→32, k=3) → Conv(32→1, k=1) → Softplus
    """

    def __init__(
        self,
        n: int = 64,
        n_features: int = 1,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        # Encoder
        self.enc1 = ConvBNrelu(n_features, n, kernel_size=15, p=dropout_p)
        self.enc2 = ResConvBNrelu(n, n, kernel_size=5, p=dropout_p)
        self.enc3 = ResConvBNrelu(n, n, kernel_size=5, p=dropout_p)
        self.enc4 = ResConvBNrelu(n, n, kernel_size=5, p=dropout_p)
        self.pool = nn.MaxPool1d(2)

        # Decoder — interpolation upsampling, not ConvTranspose
        self.up1 = InterpUp(n, n, kernel_size=3, p=dropout_p)
        self.dec1 = ResConvBNrelu(n, n, kernel_size=5, p=dropout_p)
        self.up2 = InterpUp(n, n, kernel_size=3, p=dropout_p)
        self.dec2 = ResConvBNrelu(n, n, kernel_size=5, p=dropout_p)

        # Output head — pointwise final conv prevents neighbor correlation
        self.head = nn.Sequential(
            ConvBNrelu(n, n, kernel_size=5, p=dropout_p),
            ConvBNrelu(n, n // 2, kernel_size=3, p=dropout_p),
            nn.Conv1d(n // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)  # (B, n, 1000)
        e2 = self.enc2(e1)  # (B, n, 1000)
        e3 = self.enc3(self.pool(e2))  # (B, n, 500)
        e4 = self.enc4(self.pool(e3))  # (B, n, 250)  bottleneck

        # Decoder with additive skip connections
        d1 = self.dec1(self.up1(e4) + e3)  # (B, n, 500)
        d2 = self.dec2(self.up2(d1) + e2)  # (B, n, 1000)

        # Output
        return F.softplus(self.head(d2)).squeeze(1)

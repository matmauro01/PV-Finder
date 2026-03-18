"""UNet_1000_v2 — sidelobe-free K2H architecture.

Replaces ConvTranspose1d upsampling with nearest-neighbor interpolation + Conv1d,
reduces bottleneck from 8x to 4x, and uses additive skip connections with
residual encoder blocks. See docs/models/vertex_finding.md for rationale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pv_finder.models.autoencoder_models import ConvBNrelu, MaskedDNN, ResConvBNrelu


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


class TracksToHist_v2(nn.Module):
    """End-to-end tracks → histogram model composing MaskedDNN + UNet_1000_v2.

    Unlike trackstoHists_UNet_1000 (which duplicates both architectures inline),
    this class wraps the standalone models as submodules. Weight loading is trivial:
    load each submodule's checkpoint into model.t2kde / model.k2h directly.
    """

    def __init__(self, t2kde: MaskedDNN, k2h: UNet_1000_v2):
        super().__init__()
        self.t2kde = t2kde
        self.k2h = k2h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kde = self.t2kde(x)  # (B, 1000)
        return self.k2h(kde.unsqueeze(1))  # (B, 1, 1000) → (B, 1000)

    @staticmethod
    def from_checkpoints(
        t2kde_path: str,
        k2h_path: str,
        device: torch.device = torch.device("cpu"),
    ) -> "TracksToHist_v2":
        """Build and load from separate T2KDE and K2H v2 checkpoints."""
        t2kde = MaskedDNN(
            input_size=7,
            hidden_nodes=[100, 100, 100, 100, 100],
            output_size=1000,
            leaky_param=0.01,
            maskVal=-240.0,
            predScaleFactor=0.001,
        )
        k2h = UNet_1000_v2(n=64, n_features=1, dropout_p=0.0)

        for path, mod, label in [
            (t2kde_path, t2kde, "T2KDE"),
            (k2h_path, k2h, "K2H_v2"),
        ]:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            state = (
                ckpt["model_state"]
                if isinstance(ckpt, dict) and "model_state" in ckpt
                else ckpt.state_dict()
                if hasattr(ckpt, "state_dict")
                else ckpt
            )
            mod.load_state_dict(state)
            n = sum(p.numel() for p in mod.parameters())
            print(f"  {label}: loaded {path} ({n:,} params)")

        model = TracksToHist_v2(t2kde, k2h)
        return model.to(device)

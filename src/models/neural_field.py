import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualNeuralFieldBanded(nn.Module):
    """
    Per-subject residual neural field for banded HRTF magnitudes.

    For inputs (az_deg, ear_idx), predicts:
        Δm_band[ear, band] in dB

    Inputs:
        az_deg:  (N,) float32, in degrees
        ear_idx: (N,) int64, 0 = left, 1 = right

    Encodes:
        [sin(az), cos(az), ear_scalar]
        ear_scalar = -1 (L) or +1 (R)

    Constructor:
        n_bands:    number of magnitude bands
        hidden_dim: MLP width
        num_layers: total linear layers (>= 2)
    """
    def __init__(
        self,
        n_bands: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        activation: str = "silu",
    ):
        super().__init__()
        self.n_bands = n_bands

        self.act = {
            "relu": F.relu,
            "silu": F.silu,
            "gelu": F.gelu,
        }[activation]

        # sin(az), cos(az), ear_scalar
        in_dim = 2 + 1

        layers = []
        last = in_dim
        for _ in range(num_layers - 1):
            layer = nn.Linear(last, hidden_dim)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            last = hidden_dim
        self.mlp = nn.ModuleList(layers)

        self.out = nn.Linear(last, n_bands)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def encode(self, az_deg: torch.Tensor, ear_idx: torch.Tensor) -> torch.Tensor:
        """
        az_deg:  (N,)
        ear_idx: (N,) int64, 0 or 1
        """
        az = az_deg * math.pi / 180.0
        sin_az = torch.sin(az)
        cos_az = torch.cos(az)

        ear_scalar = torch.where(
            ear_idx == 0,
            torch.full_like(ear_idx, -1.0, dtype=torch.float32),
            torch.full_like(ear_idx, +1.0, dtype=torch.float32),
        )

        return torch.stack([sin_az, cos_az, ear_scalar], dim=-1)

    def forward(self, az_deg: torch.Tensor, ear_idx: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            residual_bands: (N, n_bands)
        """
        x = self.encode(az_deg, ear_idx)
        for layer in self.mlp:
            x = self.act(layer(x))
        return self.out(x)


class GlobalResidualNeuralFieldBanded(nn.Module):
    """
    Global residual neural field with subject embeddings.

    For inputs (az_deg, ear_idx, subj_idx), predicts:
        Δm_band[ear, band] in dB

    Encodes:
        [sin(az), cos(az), ear_scalar, subject_embedding]
    """
    def __init__(
        self,
        n_bands: int,
        num_subjects: int,
        emb_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 4,
        activation: str = "silu",
    ):
        super().__init__()
        self.n_bands = n_bands
        self.num_subjects = num_subjects
        self.emb = nn.Embedding(num_subjects, emb_dim)

        self.act = {
            "relu": F.relu,
            "silu": F.silu,
            "gelu": F.gelu,
        }[activation]

        in_dim = 2 + 1 + emb_dim  # sin, cos, ear_scalar, subj_emb

        layers = []
        last = in_dim
        for _ in range(num_layers - 1):
            layer = nn.Linear(last, hidden_dim)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            last = hidden_dim
        self.mlp = nn.ModuleList(layers)

        self.out = nn.Linear(last, n_bands)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        nn.init.normal_(self.emb.weight, mean=0.0, std=0.05)

    def encode(self, az_deg, ear_idx, subj_idx):
        az = az_deg * math.pi / 180.0
        sin_az = torch.sin(az)
        cos_az = torch.cos(az)

        ear_scalar = torch.where(
            ear_idx == 0,
            torch.full_like(ear_idx, -1.0, dtype=torch.float32),
            torch.full_like(ear_idx, +1.0, dtype=torch.float32),
        )

        emb = self.emb(subj_idx)
        base = torch.stack([sin_az, cos_az, ear_scalar], dim=-1)
        return torch.cat([base, emb], dim=-1)

    def forward(self, az_deg, ear_idx, subj_idx):
        """
        Returns:
            residual_bands: (N, n_bands)
        """
        x = self.encode(az_deg, ear_idx, subj_idx)
        for layer in self.mlp:
            x = self.act(layer(x))
        return self.out(x)

import math

import torch
import torch.nn as nn

from ..config import config


class HestonMLP(nn.Module):
    def __init__(
        self,
        n_points: int = config.N_GRID,
        in_channels: int = 3,
        hidden=(512, 256, 128),
        out_dim: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_points = n_points
        layers, prev = [], n_points * in_channels
        for h in hidden:

            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))


def sinusoidal_pe(n_positions: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(n_positions, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model))
    pe = torch.zeros(n_positions, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


class HestonTransformerPE(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        n_points: int = config.N_GRID,
        out_dim: int = 5,
        decoder_hidden=(256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Linear(3, d_model)
        self.register_buffer("pe", sinusoidal_pe(n_points, d_model), persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        layers, prev = [], d_model
        for h in decoder_hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x) + self.pe[:, : x.shape[1], :]
        h = self.encoder(h)
        return self.decoder(h.mean(dim=1))


def tokens_to_image(x: torch.Tensor) -> torch.Tensor:
    B = x.shape[0]
    g = x.reshape(B, config.N_TAU, config.N_LOGM, 3)
    return g.permute(0, 3, 2, 1).contiguous()


class HestonCNN2D(nn.Module):
    def __init__(self, out_dim: int = 5, dropout: float = 0.1):
        super().__init__()
        def block(cin, cout, stride):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(cout),
                nn.GELU(),
            )

        self.features = nn.Sequential(
            block(3, 32, stride=1),
            block(32, 64, stride=2),
            block(64, 128, stride=2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(tokens_to_image(x)))

import torch
import torch.nn as nn


class SAB(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + a)
        x = self.ln2(x + self.ff(x))
        return x


class PMA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_seeds: int, ff_mult: int = 4):
        super().__init__()
        self.n_seeds = n_seeds
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, d_model) / d_model**0.5)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self._attend(x, need_weights=False)
        return out

    def forward_with_attn(self, x: torch.Tensor):
        return self._attend(x, need_weights=True)

    def _attend(self, x: torch.Tensor, need_weights: bool):
        B = x.shape[0]
        s = self.seeds.expand(B, -1, -1)
        out, w = self.attn(
            s, x, x,
            need_weights=need_weights,
            average_attn_weights=False if need_weights else True,
        )
        out = self.ln1(s + out)
        out = self.ln2(out + self.ff(out))
        return out, w


class HestonSetAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_sab_layers: int = 2,
        n_seeds: int = 4,
        pooling: str = "pma",
        out_dim: int = 5,
        decoder_hidden=(128, 64),
        ff_mult: int = 2,
        dropout: float = 0.1,
    ):

        super().__init__()
        if pooling not in ("pma", "mean"):
            raise ValueError(f"unknown pooling: {pooling}")
        self.pooling = pooling
        self.n_seeds = n_seeds if pooling == "pma" else 1
        self.d_model = d_model

        self.embed = nn.Linear(3, d_model)
        self.sabs = nn.ModuleList(
            SAB(d_model, n_heads, ff_mult=ff_mult) for _ in range(n_sab_layers))
        self.pma = (PMA(d_model, n_heads, n_seeds, ff_mult=ff_mult)
                    if pooling == "pma" else None)

        dec_in = self.n_seeds * d_model if pooling == "pma" else d_model
        layers, prev = [], dec_in
        for h in decoder_hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.decoder = nn.Sequential(*layers)

    def _encode(self, x: torch.Tensor):
        """embed + SABs; returns list [embed, sab1, sab2, ...] of (B, N, d)."""
        feats = [self.embed(x)]
        for sab in self.sabs:
            feats.append(sab(feats[-1]))
        return feats

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, N, d) -> (B, n_seeds, d) pooled representation."""
        if self.pooling == "pma":
            return self.pma(tokens)
        return tokens.mean(dim=1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._encode(x)[-1]
        pooled = self._pool(tokens)
        return self.decoder(pooled.flatten(1))


    @torch.no_grad()
    def forward_with_internals(self, x: torch.Tensor):
        feats = self._encode(x)
        internals = {"embed": feats[0]}
        for i, f in enumerate(feats[1:], start=1):
            internals[f"sab{i}"] = f
        tokens = feats[-1]
        if self.pooling == "pma":
            pooled, attn = self.pma.forward_with_attn(tokens)
            internals["pma_attn"] = attn
        else:
            pooled = tokens.mean(dim=1, keepdim=True)
        internals["pooled"] = pooled
        out = self.decoder(pooled.flatten(1))
        return out, internals

    @torch.no_grad()
    def forward_seed_ablated(self, x: torch.Tensor, seed_idx: int) -> torch.Tensor:
        tokens = self._encode(x)[-1]
        pooled = self._pool(tokens)
        if seed_idx is not None:
            if not (0 <= seed_idx < pooled.shape[1]):
                raise IndexError(f"seed_idx {seed_idx} out of range for {pooled.shape[1]} seeds")
            pooled = pooled.clone()
            pooled[:, seed_idx, :] = 0.0
        return self.decoder(pooled.flatten(1))

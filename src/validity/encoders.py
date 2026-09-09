"""Sequence encoders over the deployment-state context window.

**The encoder is infrastructure, not contribution.** `NOVELTY_DECISION.md` section 4 item 8 is
explicit that no novelty may be attributed to the choice of GRU / TCN / Transformer / SSM. They are
compared on equal terms in the ablation so the paper can report encoder sensitivity as a measured
fact, and the expectation recorded in advance is that the differences are small.

All encoders map `(B, L, d) -> (B, hidden)`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ENCODERS", "build_encoder", "GRUEncoder", "TCNEncoder", "TransformerEncoderModel"]

ENCODERS: tuple[str, ...] = ("gru", "tcn", "transformer")


class GRUEncoder(nn.Module):
    """Default. Cheap, stable on short sequences, and adequate for L in the tens."""

    def __init__(self, d_in: int, hidden: int = 64, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.norm(h[-1])


class TCNEncoder(nn.Module):
    """Dilated causal convolutions. Causality is enforced by left-padding then trimming the right,
    so no position can see its own future even inside the encoder."""

    def __init__(self, d_in: int, hidden: int = 64, levels: int = 3, kernel: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        blocks: list[nn.Module] = []
        c_in = d_in
        for i in range(levels):
            blocks.append(_CausalBlock(c_in, hidden, kernel, dilation=2 ** i, dropout=dropout))
            c_in = hidden
        self.net = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(hidden)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x.transpose(1, 2))       # (B, C, L)
        return self.norm(h[:, :, -1])         # last (most recent) position


class _CausalBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = nn.functional.pad(x, (self.pad, 0))
        h = self.drop(self.act(self.conv(h)))
        return h + self.res(x)


class TransformerEncoderModel(nn.Module):
    """Small encoder stack with learned positional embeddings and mean pooling."""

    def __init__(self, d_in: int, hidden: int = 64, heads: int = 4, layers: int = 2,
                 dropout: float = 0.1, max_len: int = 256):
        super().__init__()
        self.proj = nn.Linear(d_in, hidden)
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=4 * hidden,
                                           dropout=dropout, batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x) + self.pos[:, : x.shape[1]]
        return self.norm(self.enc(h).mean(dim=1))


def build_encoder(name: str, d_in: int, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "gru":
        return GRUEncoder(d_in, **kwargs)
    if name == "tcn":
        return TCNEncoder(d_in, **kwargs)
    if name == "transformer":
        return TransformerEncoderModel(d_in, **kwargs)
    raise ValueError(f"unknown encoder {name!r}; expected one of {ENCODERS}")

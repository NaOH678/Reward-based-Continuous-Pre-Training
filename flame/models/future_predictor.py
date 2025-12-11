# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class FuturePredictorHead(nn.Module):
    """
    Map current hidden states h_t into a space aligned with future summaries F_t.

    head_type:
        - linear: LN -> Linear -> Dropout -> L2 normalize
        - mlp:    LN -> Linear -> GELU -> Dropout -> Linear -> Dropout -> L2 normalize
        - gated:  LN -> gate(Linear) -> MLP, apply sigmoid gate on residual, then Dropout -> L2 normalize
    """

    def __init__(
        self,
        hidden_size: int,
        head_type: str = "linear",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.head_type = head_type
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        if head_type == "linear":
            self.proj = nn.Linear(hidden_size, hidden_size)
        elif head_type == "mlp":
            self.ffn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
            )
        elif head_type == "gated":
            self.gate = nn.Linear(hidden_size, hidden_size)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, H) current hidden states.
        Returns:
            normalized predicted future embedding h'_t: (B, T, H)
        """
        x_norm = self.ln(x)
        if self.head_type == "linear":
            h = self.proj(x_norm)
        elif self.head_type == "mlp":
            h = self.ffn(x_norm)
        else:  # gated
            gate = torch.sigmoid(self.gate(x_norm))
            h = x + gate * self.ffn(x_norm)
        h = self.dropout(h)
        return F.normalize(h, p=2, dim=-1)

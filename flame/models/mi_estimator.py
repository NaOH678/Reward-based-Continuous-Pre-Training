# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

class MIEstimator(nn.Module):
    """
    Base class for Mutual Information Estimator.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class InfoNCEEstimator(MIEstimator):
    """
    Estimates Mutual Information using InfoNCE loss.
    """
    def __init__(self, hidden_size: int, temperature: float = 0.1):
        super().__init__(hidden_size)
        self.temperature = temperature

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (Batch, SeqLen, HiddenDim) - Current states
            y: (Batch, SeqLen, HiddenDim) - Future summaries
            valid_mask: (Batch, SeqLen) bool mask indicating positions with at least one future token.
        Returns:
            loss: scalar tensor
        """
        if valid_mask is None:
            mask = (y.abs().sum(dim=-1) > 1e-6)  # heuristic fallback
        else:
            mask = valid_mask

        x_valid = x[mask]  # (N, Hidden)
        y_valid = y[mask]  # (N, Hidden)

        if x_valid.shape[0] == 0:
            zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            return zero.requires_grad_(True)

        # Normalize
        x_norm = F.normalize(x_valid, p=2, dim=-1)
        y_norm = F.normalize(y_valid, p=2, dim=-1)

        # Align dtypes to avoid matmul failures under mixed precision.
        common_dtype = torch.promote_types(x_norm.dtype, y_norm.dtype)
        x_norm = x_norm.to(common_dtype)
        y_norm = y_norm.to(common_dtype)

        logits = torch.mm(x_norm, y_norm.t()) / self.temperature

        labels = torch.arange(x_valid.shape[0], device=x.device)

        loss = F.cross_entropy(logits, labels)

        return loss

def build_mi_estimator(estimator_type: str, hidden_size: int, **kwargs):
    if estimator_type == "infonce":
        return InfoNCEEstimator(hidden_size, **kwargs)
    else:
        raise ValueError(f"Unknown estimator type: {estimator_type}")

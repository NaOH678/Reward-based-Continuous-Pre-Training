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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, SeqLen, HiddenDim) - Current states
            y: (Batch, SeqLen, HiddenDim) - Future summaries
        Returns:
            loss: scalar tensor
            log_N: scalar tensor (log of batch size for contrastive learning)
        """
        # We only compute loss for steps where y is valid (non-zero).
        # Assuming y is zero where there is no future (e.g. last step).
        # Or we can pass a mask. For now, let's assume the caller handles masking or we ignore zeros.
        # Actually, exact zero might be a valid summary (unlikely).
        # Let's compute for all, but we need to know which steps are valid.
        # The FutureEncoder returns zeros for invalid steps.
        
        # Flatten batch and seq_len for contrastive learning?
        # Or keep batch structure?
        # Standard InfoNCE: Positive pair (x_i, y_i). Negatives: (x_i, y_j).
        
        # Let's flatten to (Batch*SeqLen, Hidden) but filter out invalid steps.
        # Invalid steps are those where y is all zeros (heuristic).
        
        mask = (y.abs().sum(dim=-1) > 1e-6) # (Batch, SeqLen)
        
        x_valid = x[mask] # (N, Hidden)
        y_valid = y[mask] # (N, Hidden)
        
        if x_valid.shape[0] == 0:
            return torch.tensor(0.0, device=x.device, requires_grad=True), torch.tensor(0.0, device=x.device)
            
        # Normalize
        x_norm = F.normalize(x_valid, p=2, dim=-1)
        y_norm = F.normalize(y_valid, p=2, dim=-1)
        
        # Similarity
        # (N, Hidden) @ (Hidden, N) -> (N, N)
        logits = torch.mm(x_norm, y_norm.t()) / self.temperature
        
        labels = torch.arange(x_valid.shape[0], device=x.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss, torch.log(torch.tensor(x_valid.shape[0], dtype=torch.float, device=x.device))

def build_mi_estimator(estimator_type: str, hidden_size: int, **kwargs):
    if estimator_type == "infonce":
        return InfoNCEEstimator(hidden_size, **kwargs)
    else:
        raise ValueError(f"Unknown estimator type: {estimator_type}")

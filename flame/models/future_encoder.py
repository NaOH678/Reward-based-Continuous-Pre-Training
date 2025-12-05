# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

class FutureEncoder(nn.Module):
    """
    Future Encoder module for Reward-based Continuous Pre-Training.
    Responsible for summarizing future states.
    """
    def __init__(
        self,
        hidden_size: int,
        future_k: int = 4,
        summary_method: str = "mean",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.future_k = future_k
        self.summary_method = summary_method

        if self.summary_method == "attention":
            # Simple self-attention style summary where we attend to future tokens.
            self.attention_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (Batch, SeqLen, HiddenDim)
        Returns:
            future_summaries: (Batch, SeqLen, HiddenDim)
            The summary at index t corresponds to the summary of the future of t.
            If no future exists (last steps), the summary might be zero or masked later.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Normalize hidden states? 
        # Usually better to normalize before summary if using attention, 
        # but for mean/max it might not matter as much. 
        # Let's keep it raw here and let the estimator handle normalization if needed,
        # OR normalize here for consistency with the previous implementation.
        # The previous implementation normalized for cosine similarity.
        # Let's return raw summaries and let the estimator decide how to compare.
        
        # However, for "attention", we need to compute scores.
        
        # To be efficient and support flexible k, we can use a loop or unfolding.
        # Since we need to return a tensor of the same shape (Batch, SeqLen, HiddenDim),
        # we will fill it with summaries.
        
        future_summaries = torch.zeros_like(hidden_states)
        
        # We can optimize this later. For now, the loop is clear.
        # Note: This loop is slow for long sequences in Python.
        # A better approach for "mean" with fixed k is AvgPool1d.
        
        if self.future_k > 0 and self.summary_method == "mean":
            # Use AvgPool1d
            # Input: (Batch, Hidden, SeqLen)
            x = hidden_states.transpose(1, 2)
            # We want window [t+1, t+k+1]. 
            # Padding: we need to handle boundaries.
            # Let's stick to the loop for correctness with the specific "future" definition first,
            # or use a mask.
            pass

        # Let's stick to the loop for now to ensure exact logic match with previous version,
        # but we need to be careful about performance.
        
        # Optimization: Pre-compute all windows? No, too much memory.
        
        for t in range(seq_len - 1):
            # Future window
            start_future = t + 1
            if self.future_k > 0:
                end_future = min(t + 1 + self.future_k, seq_len)
            else:
                end_future = seq_len
                
            if start_future >= end_future:
                continue
                
            future_window = hidden_states[:, start_future:end_future, :] # (Batch, Window, Hidden)
            
            # Compute summary
            if self.summary_method == "mean":
                summary = future_window.mean(dim=1) # (Batch, Hidden)
            elif self.summary_method == "max":
                summary = future_window.max(dim=1)[0] # (Batch, Hidden)
            elif self.summary_method == "attention":
                h_t = hidden_states[:, t, :]
                # (Batch, 1, Hidden) @ (Batch, Hidden, Window) -> (Batch, 1, Window)
                attn_scores = torch.bmm(h_t.unsqueeze(1), future_window.transpose(1, 2))
                attn_weights = F.softmax(attn_scores, dim=-1)
                summary = torch.bmm(attn_weights, future_window).squeeze(1) # (Batch, Hidden)
            else:
                raise ValueError(f"Unknown summary method: {self.summary_method}")
            
            future_summaries[:, t, :] = summary
            
        return future_summaries

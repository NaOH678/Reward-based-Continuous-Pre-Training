# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

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
            # Simple projection for query/key/value before attending to future tokens.
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (Batch, SeqLen, HiddenDim)
            attention_mask: (Batch, SeqLen) with 1 for valid tokens, 0 for padding.
        Returns:
            future_summaries: (Batch, SeqLen, HiddenDim)
            valid_mask: (Batch, SeqLen) bool, True where summary is based on at least one future token.
            The summary at index t corresponds to the summary of the future of t.
            If no future exists (last steps), the summary might be zero or masked later.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Build mask tensor
        if attention_mask is None:
            mask = hidden_states.new_ones((batch_size, seq_len), dtype=torch.bool)
        else:
            mask = attention_mask.to(dtype=torch.bool)

        # Vectorized implementation

        # 1. Mean Pooling
        if self.summary_method == "mean":
            # Use masked cumsum for efficient window sum
            masked_states = hidden_states * mask.unsqueeze(-1)
            cumsum = torch.cumsum(masked_states, dim=1)
            cumsum = torch.cat(
                [hidden_states.new_zeros((batch_size, 1, hidden_dim)), cumsum], dim=1
            )

            count_cumsum = torch.cumsum(mask.float(), dim=1)
            count_cumsum = torch.cat(
                [hidden_states.new_zeros((batch_size, 1)), count_cumsum], dim=1
            )

            indices = torch.arange(seq_len, device=hidden_states.device)
            start_indices = indices + 1

            if self.future_k > 0:
                end_indices = torch.clamp(indices + 1 + self.future_k, max=seq_len)
            else:
                end_indices = torch.full_like(indices, seq_len)

            # Handle cases where start >= end (last step)
            valid_mask = start_indices < end_indices

            S_end = cumsum[:, end_indices, :]
            S_start = cumsum[:, start_indices, :]

            window_sum = S_end - S_start
            count_end = count_cumsum[:, end_indices]
            count_start = count_cumsum[:, start_indices]
            window_count = count_end - count_start

            # Avoid division by zero
            window_count_clamped = torch.clamp(window_count, min=1.0)

            future_summaries = window_sum / window_count_clamped.unsqueeze(-1)

            # Mask out invalid steps (where no future exists)
            future_valid = valid_mask.unsqueeze(0) & (window_count > 0).unsqueeze(0)
            future_summaries = torch.where(
                future_valid.unsqueeze(-1), future_summaries, hidden_states.new_zeros(1)
            )

            return future_summaries, future_valid

        # 2. Max Pooling
        elif self.summary_method == "max":
            if self.future_k == -1:
                # Global max from t+1 to end, mask padding as very negative.
                very_neg = torch.finfo(hidden_states.dtype).min
                masked_states = hidden_states.masked_fill(~mask.unsqueeze(-1), very_neg)
                rev_states = masked_states.flip(dims=[1])
                cummax_rev = rev_states.cummax(dim=1)[0]
                global_future_max = cummax_rev.flip(dims=[1])

                # Shift left to get max starting from t+1
                future_summaries = torch.zeros_like(hidden_states)
                future_summaries[:, :-1, :] = global_future_max[:, 1:, :]
                future_valid = mask[:, None, :].expand(-1, seq_len, -1).any(dim=-1)
                return future_summaries, future_valid
            else:
                # Sliding window max; pad with very negative to ignore invalid.
                very_neg = torch.finfo(hidden_states.dtype).min
                x = hidden_states.transpose(1, 2)
                pad_size = self.future_k
                mask_padded = F.pad(mask, (0, pad_size), value=False)
                future_mask = mask_padded.unfold(1, self.future_k, 1)[:, 1 : seq_len + 1, :]
                x_padded = F.pad(x[:, :, 1:], (0, pad_size), value=very_neg)
                pooled = F.max_pool1d(x_padded, kernel_size=self.future_k, stride=1)
                future_summaries = pooled[:, :, :seq_len].transpose(1, 2)
                future_valid = future_mask.any(dim=-1)
                future_summaries = torch.where(
                    future_valid.unsqueeze(-1), future_summaries, hidden_states.new_zeros(1)
                )
                return future_summaries, future_valid

        # 3. Attention
        elif self.summary_method == "attention":
            if self.future_k == -1:
                q = self.q_proj(hidden_states)
                k = self.k_proj(hidden_states)
                v = self.v_proj(hidden_states)

                attn_scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(hidden_dim)

                future_only = torch.triu(
                    hidden_states.new_ones((seq_len, seq_len), dtype=torch.bool),
                    diagonal=1,
                )
                # mask padding on keys
                key_mask = mask.unsqueeze(1)  # (B,1,T)
                combined_mask = future_only.unsqueeze(0) & key_mask
                attn_scores.masked_fill_(~combined_mask, float("-inf"))

                attn_weights = F.softmax(attn_scores, dim=-1)
                attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
                summary = torch.bmm(attn_weights, v)

                # valid if there exists a future valid token
                future_valid = combined_mask.any(dim=-1)
                return summary, future_valid
            else:
                # Sliding window attention
                pad_size = self.future_k
                # Pad time dim
                x_padded = F.pad(hidden_states, (0, 0, 0, pad_size))
                mask_padded = F.pad(mask, (0, pad_size), value=False)

                windows = x_padded.unfold(1, self.future_k, 1)
                mask_windows = mask_padded.unfold(1, self.future_k, 1)
                future_windows = windows[:, 1 : seq_len + 1, :, :]  # (B, T, k, D)
                future_mask = mask_windows[:, 1 : seq_len + 1, :]  # (B, T, k)

                q = self.q_proj(hidden_states).unsqueeze(2)
                k = self.k_proj(future_windows)
                v = self.v_proj(future_windows)

                attn_scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(hidden_dim)

                attn_scores.masked_fill_(~future_mask.unsqueeze(2), float("-inf"))

                attn_weights = F.softmax(attn_scores, dim=-1)
                attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

                summary = torch.matmul(attn_weights, v).squeeze(2)
                future_valid = future_mask.any(dim=-1)

                return summary, future_valid

        return hidden_states.new_zeros(hidden_states.shape), mask.new_zeros(mask.shape)

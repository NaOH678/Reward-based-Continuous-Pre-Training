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
        # Usually better to normalize before usmmary if using attention, 
        # but for mean/max it might not matter as much. 
        # Let's keep it raw here and let the estimator handle normalization if needed,
        # OR normalize here for consistency with the previous implementation.
        # The previous implementation normalized for cosine similarity.
        # Let's return raw summaries and let the estimator decide how to compare.
        
        # However, for "attention", we need to compute scores.
        
        # To be efficient and support flexible k, we can use a loop or unfolding.
        # Since we need to return a tensor of the same shape (Batch, SeqLen, HiddenDim),
        # we will fill it with summaries.
        
        # Vectorized implementation
        
        # 1. Mean Pooling
        if self.summary_method == "mean":
            # Use cumsum for O(1) window sum calculation
            # S[i] = sum(0...i)
            # Sum(i...j) = S[j] - S[i-1]
            
            # We want sum from t+1 to min(t+1+k, T)
            # Let's pad cumsum with a zero at the beginning for easy indexing
            cumsum = torch.cumsum(hidden_states, dim=1)
            cumsum = torch.cat([torch.zeros(batch_size, 1, hidden_dim, device=hidden_states.device), cumsum], dim=1)
            
            indices = torch.arange(seq_len, device=hidden_states.device)
            start_indices = indices + 1
            
            if self.future_k > 0:
                end_indices = torch.clamp(indices + 1 + self.future_k, max=seq_len)
            else:
                end_indices = torch.full_like(indices, seq_len)
                
            # Handle cases where start >= end (last step)
            valid_mask = start_indices < end_indices
            
            # Gather sums
            # (Batch, SeqLen, Hidden)
            # We need to expand indices to match batch size or use broadcasting?
            # cumsum is (B, T+1, D). We want to select along T dim.
            # S_end = cumsum[:, end_indices]
            # S_start = cumsum[:, start_indices]
            
            # end_indices is (T,). We can broadcast.
            S_end = cumsum[:, end_indices, :]
            S_start = cumsum[:, start_indices, :]
            
            window_sum = S_end - S_start
            window_count = (end_indices - start_indices).unsqueeze(-1).unsqueeze(0).float()
            
            # Avoid division by zero
            window_count = torch.clamp(window_count, min=1.0)
            
            future_summaries = window_sum / window_count
            
            # Mask out invalid steps (where no future exists)
            future_summaries[~valid_mask.unsqueeze(0).unsqueeze(-1).expand_as(future_summaries)] = 0.0
            
            return future_summaries

        # 2. Max Pooling
        elif self.summary_method == "max":
            if self.future_k == -1:
                # Global max from t+1 to end
                # Reverse cummax
                # x_rev = x.flip(1)
                # cummax_rev = x_rev.cummax(1)[0]
                # future_max = cummax_rev.flip(1)
                # This gives max(t...end). We want max(t+1...end).
                # So future_max[t] = cummax_rev_flip[t+1]
                
                rev_states = hidden_states.flip(dims=[1])
                cummax_rev = rev_states.cummax(dim=1)[0]
                global_future_max = cummax_rev.flip(dims=[1])
                
                # Shift left to get max starting from t+1
                future_summaries = torch.zeros_like(hidden_states)
                future_summaries[:, :-1, :] = global_future_max[:, 1:, :]
                return future_summaries
            else:
                # Sliding window max
                # Use MaxPool1d
                # Input: (B, D, T)
                x = hidden_states.transpose(1, 2)
                # We want window [t+1, t+k+1].
                # MaxPool1d(kernel=k, stride=1) gives windows [t, t+k].
                # If we apply to x[:, :, 1:], we get windows starting at t+1.
                # Padding is needed for the end.
                
                # Pad with -inf to handle boundaries correctly
                pad_size = self.future_k
                x_padded = F.pad(x[:, :, 1:], (0, pad_size), value=-1e9)
                
                # (B, D, T)
                pooled = F.max_pool1d(x_padded, kernel_size=self.future_k, stride=1)
                
                # Crop to original length
                future_summaries = pooled[:, :, :seq_len].transpose(1, 2)
                return future_summaries

        # 3. Attention
        elif self.summary_method == "attention":
            if self.future_k == -1:
                # All future attention
                # We can use standard attention with a mask.
                # But we want to attend ONLY to future.
                # Standard causal mask allows attending to past.
                # We want inverted causal mask: attend to j > i.
                
                # Query: h_t
                # Key/Value: h_all
                # Mask: M[i, j] = 0 if j > i else -inf
                
                # (B, 1, T, D)
                q = hidden_states
                k = hidden_states
                v = hidden_states
                
                # Scaled Dot Product Attention
                # (B, T, D) x (B, D, T) -> (B, T, T)
                attn_scores = torch.bmm(q, k.transpose(1, 2))
                
                # Mask
                ones = torch.ones(seq_len, seq_len, device=hidden_states.device)
                # We want to keep j > i.
                # triu(diagonal=1) gives upper triangle (j > i).
                mask = torch.triu(ones, diagonal=1).bool()
                
                attn_scores.masked_fill_(~mask.unsqueeze(0), float("-inf"))
                
                attn_weights = F.softmax(attn_scores, dim=-1)
                
                # Handle rows with all -inf (last step) -> NaN after softmax
                # Replace NaNs with 0
                attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
                
                summary = torch.bmm(attn_weights, v)
                return summary
            else:
                # Sliding window attention
                # Use unfold
                # Pad hidden_states with zeros at the end
                # We want windows of size k starting at t+1
                
                pad_size = self.future_k
                # Pad time dim
                x_padded = F.pad(hidden_states, (0, 0, 0, pad_size))
                
                # Unfold: (B, T_windows, k, D)
                # We want T windows.
                # unfold(1, k, 1) gives T_padded - k + 1 windows.
                # T + k - k + 1 = T + 1 windows.
                windows = x_padded.unfold(1, self.future_k, 1)
                
                # We want window at t to be x[t+1 : t+1+k]
                # windows[i] is x[i : i+k]
                # So we want windows[1 : T+1]
                # But x_padded has length T+k.
                # windows has length T+1.
                # windows[1] corresponds to x[1:1+k] -> future of t=0. Correct.
                # windows[T] corresponds to x[T:T+k] -> future of t=T-1. Correct.
                
                future_windows = windows[:, 1:seq_len+1, :, :] # (B, T, k, D)
                
                # Query: h_t (B, T, D) -> (B, T, 1, D)
                q = hidden_states.unsqueeze(2)
                
                # Attention
                # (B, T, 1, D) @ (B, T, D, k) -> (B, T, 1, k)
                attn_scores = torch.matmul(q, future_windows.transpose(2, 3))
                
                # Mask padding?
                # If we are near the end, the window contains padded zeros.
                # We should mask them out.
                # Or, since they are zeros, their dot product with q will be 0.
                # But softmax(0, 0, 0) is uniform, which is wrong. We want to ignore them.
                # We need a mask for the padding.
                
                # Construct mask
                # Valid indices for each t: [t+1, min(t+1+k, T)]
                # In the window frame (0 to k-1): valid if t+1+j < T
                
                indices = torch.arange(seq_len, device=hidden_states.device).unsqueeze(-1) # (T, 1)
                window_indices = torch.arange(self.future_k, device=hidden_states.device).unsqueeze(0) # (1, k)
                # Global index of token in window: t + 1 + j
                global_indices = indices + 1 + window_indices # (T, k)
                
                valid_mask = global_indices < seq_len
                
                attn_scores.masked_fill_(~valid_mask.unsqueeze(0).unsqueeze(2), float("-inf"))
                
                attn_weights = F.softmax(attn_scores, dim=-1)
                attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
                
                # (B, T, 1, k) @ (B, T, k, D) -> (B, T, 1, D)
                summary = torch.matmul(attn_weights, future_windows).squeeze(2)
                
                return summary

        return future_summaries

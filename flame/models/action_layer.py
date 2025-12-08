import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Replicate

try:
    RMSNorm = nn.RMSNorm  # type: ignore[attr-defined]
except AttributeError:
    class RMSNorm(nn.Module):  # minimal fallback
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
            return (x / norm.clamp(min=self.eps)) * self.weight


class ActionLayer(nn.Module):
    """
    Token-level future-aware reward head.
    Builds an action set from student logits (top-k + ground-truth),
    scores each candidate against the future summary, and returns a policy-gradient-style loss.
    """

    def __init__(
        self,
        hidden_size: int,
        top_k: int = 16,
        head_type: str = "mlp",
        tau: float = 1.0,
        reward_clamp: float | None = None,
        mean_threshold: bool = False,
        gt_bias: float = 0.0,
        use_rms_norm: bool = False,
        ffn_hidden_size: int | None = None,
        activation: str | None = None,
        residual: bool = False,
        delta_init_zero: bool = False,
        score_type: str = "cosine",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.tau = tau
        self.reward_clamp = reward_clamp
        self.mean_threshold = mean_threshold
        self.gt_bias = gt_bias
        self.use_rms_norm = use_rms_norm
        self.ffn_hidden_size = ffn_hidden_size
        self.activation = (activation or "gelu").lower()
        self.residual = residual
        self.delta_init_zero = delta_init_zero
        self.score_type = score_type
        # Cache for already-localized embedding weights to avoid repeated DTensor->local conversions.
        self._cached_embed_src = None
        self._cached_embed_local = None

        head_type = head_type.lower()
        self.head_type = head_type
        
        if head_type == "mlp":
            # lightweight concat+MLP to predict future summary
            inner = self.ffn_hidden_size or hidden_size * 2
            norm_cls = RMSNorm if use_rms_norm else nn.LayerNorm
            self.cos_mlp = nn.Sequential(
                nn.Linear(hidden_size * 2, inner),
                nn.GELU(),
                nn.Linear(inner, hidden_size),
                norm_cls(hidden_size),
            )
            if self.residual and self.delta_init_zero:
                last_linear = self.cos_mlp[2]
                if isinstance(last_linear, nn.Linear):
                    nn.init.zeros_(last_linear.weight)
                    if last_linear.bias is not None:
                        nn.init.zeros_(last_linear.bias)
        elif head_type == "tower":
            # dual-encoder style: project h and e separately then fuse
            proj_dim = self.ffn_hidden_size or hidden_size
            norm_cls = RMSNorm if use_rms_norm else nn.LayerNorm
            self.tower_h = nn.Linear(hidden_size, proj_dim)
            self.tower_e = nn.Linear(hidden_size, proj_dim)
            fuse_dim = proj_dim * 3  # concat h, e, h*e
            self.tower_mlp = nn.Sequential(
                nn.Linear(fuse_dim, fuse_dim),
                nn.GELU(),
                nn.Linear(fuse_dim, hidden_size),
                norm_cls(hidden_size),
            )
            if self.residual and self.delta_init_zero:
                last_linear = self.tower_mlp[2]
                if isinstance(last_linear, nn.Linear):
                    nn.init.zeros_(last_linear.weight)
                    if last_linear.bias is not None:
                        nn.init.zeros_(last_linear.bias)
        elif head_type == "bilinear":
            raise ValueError(f"Unknown head_type: {head_type}")

    @torch.no_grad()
    def _build_action_set(self, logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized construction of action indices and mask.
        Returns:
            action_indices: (B*T, max_actions)
            action_mask: (B*T, max_actions) bool
        """
        batch_time, vocab_size = logits.shape
        top_k = max(self.top_k, 0)
        if top_k > 0:
            _, topk_indices = logits.topk(top_k, dim=-1)
        else:
            topk_indices = logits.new_zeros((batch_time, 0), dtype=torch.long)

        combined = torch.cat([topk_indices, labels.unsqueeze(1)], dim=1)  # (B*T, top_k+1)
        max_actions = combined.size(1)

        sorted_idx, sort_order = combined.sort(dim=1)
        unique_mask_sorted = torch.cat(
            [
                torch.ones(batch_time, 1, device=logits.device, dtype=torch.bool),
                sorted_idx[:, 1:] != sorted_idx[:, :-1],
            ],
            dim=1,
        )
        unsort_order = sort_order.argsort(dim=1)
        unique_mask = unique_mask_sorted.gather(1, unsort_order)

        position_indices = unique_mask.cumsum(dim=1) - 1
        action_indices = torch.zeros(batch_time, max_actions, device=logits.device, dtype=torch.long)
        action_mask = torch.zeros_like(action_indices, dtype=torch.bool)

        row_idx, col_idx = unique_mask.nonzero(as_tuple=True)
        pos = position_indices[row_idx, col_idx]
        action_indices[row_idx, pos] = combined[row_idx, col_idx]
        action_mask[row_idx, pos] = True
        return action_indices, action_mask

    def forward(
        self,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        future_summaries: torch.Tensor,
        future_valid: torch.Tensor | None,
        embed_weight: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            logits: (B, T, V) student logits
            hidden_states: (B, T, H) last-layer hidden states
            labels: (B, T) target token ids (-100 for ignore)
            future_summaries: (B, T, H) future summaries
            future_valid: (B, T) bool indicating positions with valid future
            embed_weight: (V, H) token embedding matrix
            attention_mask: (B, T) optional padding mask
        Returns:
            loss: scalar tensor
            metrics: dict of torch scalars
        """
        # Avoid mixed DTensor/Tensor in embedding lookup.
        if isinstance(embed_weight, DTensor):
            if embed_weight is not self._cached_embed_src:
                if any(not isinstance(p, Replicate) for p in embed_weight.placements):
                    embed_weight = embed_weight.redistribute(
                        embed_weight.device_mesh,
                        placements=[Replicate()] * embed_weight.ndim,
                    )
                self._cached_embed_src = embed_weight
                self._cached_embed_local = embed_weight.to_local()
            embed_weight = self._cached_embed_local

        bsz, seq_len, vocab_size = logits.shape
        hidden_size = hidden_states.size(-1)
        logits_flat = logits.view(-1, vocab_size)
        hidden_flat = hidden_states.view(-1, hidden_size)
        labels_flat = labels.view(-1)
        future_flat = future_summaries.view(-1, hidden_size)

        valid_mask = labels_flat != -100
        if attention_mask is not None:
            valid_mask = valid_mask & attention_mask.view(-1).to(torch.bool)
        if future_valid is not None:
            valid_mask = valid_mask & future_valid.view(-1)

        action_indices, action_mask = self._build_action_set(logits_flat, labels_flat)

        token_embeds = F.embedding(action_indices, embed_weight)
        token_embeds = token_embeds.to(hidden_states.dtype)

        hidden_expanded = hidden_flat.unsqueeze(1).expand_as(token_embeds)
        future_expanded = future_flat.unsqueeze(1).expand_as(token_embeds)

        if self.head_type == "mlp":
            fused = torch.cat([hidden_expanded, token_embeds], dim=-1)  # (B*T, A, 2H)
            delta = self.cos_mlp(fused)  # (B*T, A, H)
            action_repr = future_expanded + delta if self.residual else delta
            scores = self._compute_scores(action_repr, future_expanded)
        elif self.head_type == "tower":
            h_proj = self.tower_h(hidden_expanded)
            e_proj = self.tower_e(token_embeds)
            sim = h_proj * e_proj
            fused = torch.cat([h_proj, e_proj, sim], dim=-1)
            delta = self.tower_mlp(fused)
            action_repr = future_expanded + delta if self.residual else delta
            scores = self._compute_scores(action_repr, future_expanded)
        else:
            raise NotImplementedError
        

        if self.gt_bias != 0.0:
            gt_mask = action_indices == labels_flat.unsqueeze(1)
            scores = scores + self.gt_bias * gt_mask

        scores = scores.masked_fill(~action_mask, -1e9)

        rewards = torch.softmax(scores / self.tau, dim=-1)
        rewards = rewards * action_mask

        if self.mean_threshold:
            valid_counts = action_mask.sum(dim=1, keepdim=True).clamp(min=1)
            mean_prob = rewards.sum(dim=1, keepdim=True) / valid_counts
            rewards = torch.where(rewards > mean_prob, rewards, torch.zeros_like(rewards))
            rewards = rewards / (rewards.sum(dim=1, keepdim=True) + 1e-8)

        if self.reward_clamp is not None:
            rewards = torch.clamp(rewards, max=self.reward_clamp)
            rewards = rewards / (rewards.sum(dim=1, keepdim=True) + 1e-8)

        action_mask = action_mask & valid_mask.unsqueeze(1)

        log_probs_full = F.log_softmax(logits_flat, dim=-1)
        log_probs_actions = log_probs_full.gather(1, action_indices) * action_mask

        loss_per_pos = -(rewards.detach() * log_probs_actions).sum(dim=1)
        denom = action_mask.sum(dim=1).clamp(min=1)
        loss_per_pos = loss_per_pos / denom
        loss_per_pos = loss_per_pos * valid_mask.float()

        normalizer = valid_mask.float().sum().clamp(min=1)
        loss = loss_per_pos.sum() / normalizer

        valid_actions = action_mask.any(dim=1)
        rewards_valid = rewards[action_mask & valid_mask.unsqueeze(1)]
        avg_reward = rewards_valid.mean() if rewards_valid.numel() > 0 else rewards.new_tensor(0.0)
        max_reward = rewards_valid.max() if rewards_valid.numel() > 0 else rewards.new_tensor(0.0)
        action_sizes = action_mask[valid_actions].sum(dim=1).float() if valid_actions.any() else action_mask.sum(dim=1).float()
        avg_action_size = action_sizes.mean() if action_sizes.numel() > 0 else loss.new_tensor(0.0)

        metrics = {
            "avg_reward": avg_reward,
            "max_reward": max_reward,
            "avg_action_size": avg_action_size,
        }
        return loss, metrics

    def _compute_scores(self, action_repr: torch.Tensor, future_expanded: torch.Tensor) -> torch.Tensor:
        if self.score_type == "cosine":
            pred_future = F.normalize(action_repr, dim=-1)
            target_future = F.normalize(future_expanded, dim=-1)
            return (pred_future * target_future).sum(dim=-1)
        elif self.score_type == "delta_l2":
            delta = action_repr - future_expanded if self.residual else action_repr
            return -delta.norm(dim=-1)
        else:
            raise ValueError(f"Unknown score_type: {self.score_type}")

    

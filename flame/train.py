# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
import os
import time
from datetime import timedelta
from contextlib import nullcontext

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datasets import IterableDataset
import fla  # noqa
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
from fla.ops.utils import prepare_position_ids
try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    _HAS_FLEX_ATTENTION = True
except ImportError:  # pragma: no cover - flex_attention not available on older torch
    _HAS_FLEX_ATTENTION = False
try:
    from transformers import AttentionInterface, AttentionMaskInterface
    _HAS_FLEX_INTERFACE = False
except Exception:  # pragma: no cover - older transformers
    _HAS_FLEX_INTERFACE = False
from torch.distributed.elastic.multiprocessing.errors import record
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.ft import FTParallelDims, init_ft_manager
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.metrics import build_device_memory_monitor, build_metrics_processor, ensure_pp_loss_visible
from torchtitan.components.optimizer import build_optimizers
from torchtitan.config_manager import TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.distributed import utils as dist_utils
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.protocols.train_spec import TrainSpec, get_train_spec, register_train_spec
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import custom_models
from flame.components.checkpoint import TrainState
from flame.config_manager import JobConfig
from flame.data import build_dataloader, build_dataset
from flame.models.parallelize_fla import apply_ddp, apply_fsdp, parallelize_fla
from flame.models.pipeline_fla import pipeline_fla
from flame.models.mi_estimator import build_mi_estimator
from flame.models.future_predictor import FuturePredictorHead
from flame.models.action_layer import ActionLayer
from flame.tools.utils import get_nparams_and_flops


def init_rope_inv_freq(model, device):
    """
    手动初始化 RoPE 的 inv_freq buffer。

    这是因为 HuggingFace 模型将 inv_freq 注册为 non-persistent buffer，
    不会保存到 checkpoint 中。当从 DCP checkpoint 加载时，post_init()
    在 to_empty() 后无法正确初始化 inv_freq，导致 RoPE 失效。

    Args:
        model: HuggingFace 模型 (LlamaForCausalLM, OLMoForCausalLM, etc.)
        device: 目标设备
    """
    config = model.config

    # 获取 RoPE 参数
    rope_theta = getattr(config, 'rope_theta', 10000.0)
    head_dim = config.hidden_size // config.num_attention_heads

    # 计算 inv_freq: 1 / (theta ^ (2i / dim))
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    inv_freq = inv_freq.to(device)

    # 找到并设置所有 rotary_emb.inv_freq buffers
    initialized = False
    for name, module in model.named_modules():
        if hasattr(module, 'inv_freq') and 'rotary' in name.lower():
            if module.inv_freq.shape == inv_freq.shape:
                module.inv_freq.copy_(inv_freq)
                initialized = True

    # 也处理顶层 rotary_emb (如 model.model.rotary_emb)
    if hasattr(model, 'model') and hasattr(model.model, 'rotary_emb'):
        rotary_emb = model.model.rotary_emb
        if hasattr(rotary_emb, 'inv_freq') and rotary_emb.inv_freq.shape == inv_freq.shape:
            rotary_emb.inv_freq.copy_(inv_freq)
            initialized = True

    if initialized:
        logger.info(f"Initialized RoPE inv_freq: rope_theta={rope_theta}, head_dim={head_dim}")
    else:
        logger.warning("Could not find inv_freq buffer to initialize - model may not use RoPE")


def _peek_raw_sample(dataset):
    """Return a single raw sample without assuming map or iterable semantics."""
    if isinstance(dataset, IterableDataset):
        iterator = dataset.take(1) if hasattr(dataset, "take") else iter(dataset)
        try:
            return next(iter(iterator))
        except StopIteration:
            return None
    try:
        return dataset[0]
    except Exception:
        return None


def _log_sample_preview(dataset, tokenizer, color, max_chars: int = 400, max_tokens: int = 64):
    sample = _peek_raw_sample(dataset)
    if not sample:
        logger.warning("Could not preview a training sample (dataset may be empty or non-iterable).")
        return

    text = sample.get("text") or sample.get("content") or ""
    text_truncated = text if len(text) <= max_chars else f"{text[:max_chars]}…"
    logger.info(f"{color.cyan}Sample text preview (truncated to {max_chars} chars):{color.reset}\n{text_truncated}")

    if tokenizer and text:
        try:
            token_ids = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
            decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
            logger.info(f"{color.cyan}First {len(token_ids)} token ids:{color.reset} {token_ids}")
            logger.info(f"{color.cyan}Decoded preview:{color.reset}\n{decoded}")
        except Exception as exc:
            logger.warning(f"Failed to tokenize preview sample: {exc}")

_MODEL_CONVERTER_HOOK_MODEL_PARTS = None
_MODEL_CONVERTERS = None
_BLOCK_MASK_CACHE: dict = {}
_WARNED_NO_FLEX = False
_FLEX_REG_DONE = False
_FLEX_DEBUG = bool(int(os.environ.get("FLEX_DEBUG", "0")))


def _optimizer_post_step_hook(*args, **kwargs):
    if _MODEL_CONVERTER_HOOK_MODEL_PARTS is None or _MODEL_CONVERTERS is None:
        return
    _MODEL_CONVERTERS.post_optimizer_hook(_MODEL_CONVERTER_HOOK_MODEL_PARTS)


def _ensure_cloudpickle_for_dist_objects():
    """Allow torch.distributed collectives to serialize complex Python objects."""

    try:
        import cloudpickle  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - optional dependency missing
        logger.warning(
            "cloudpickle is not installed; distributed checkpoint coordination may fail "
            "when planner state contains non-picklable objects."
        )
        return

    import torch.distributed.distributed_c10d as dist_c10d

    if getattr(dist_c10d, "_pickler", None) is not cloudpickle.CloudPickler:
        logger.info("Enabling cloudpickle for distributed object collectives")
        dist_c10d._pickler = cloudpickle.CloudPickler  # type: ignore[attr-defined]


def build_future_attention_mask(
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
    window_k: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a padding-aware anti-causal attention mask and a validity mask for InfoNCE.

    Args:
        attention_mask: (B, T) with 1 for valid tokens, 0 for padding.
        dtype: dtype to use for the additive mask.
        window_k: Optional window size for future attention. If None, attend to all future tokens.
    Returns:
        future_attn_mask: (B, 1, T, T) additive mask, 0 where attending is allowed, large negative elsewhere.
        future_valid: (B, T) bool mask where at least one valid future token exists.
    """
    bsz, seqlen = attention_mask.shape
    device = attention_mask.device
    neg_inf = -1e4

    # Create position indices
    positions = torch.arange(seqlen, device=device)

    # Future-only mask: allow attending to positions strictly greater than t
    # future_only[i, j] = True if j > i (anti-causal)
    future_only = positions[None, :] > positions[:, None]  # [T, T]

    # Apply window constraint if specified
    if window_k is not None and window_k > 0:
        # Also require j <= i + window_k
        distance = positions[None, :] - positions[:, None]  # j - i
        within_window = distance <= window_k
        future_only = future_only & within_window

    # Convert to additive mask
    future_mask = torch.where(
        future_only,
        0.0,
        neg_inf
    ).to(dtype=dtype)  # [T, T]

    # Expand to batch
    future_mask = future_mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, seqlen, seqlen).contiguous()

    # Apply padding mask
    if attention_mask is not None:
        pad_mask = (attention_mask == 0).to(dtype=dtype)
        future_mask = future_mask + pad_mask[:, None, None, :] * neg_inf

    # Valid positions are non-pad tokens that have at least one future valid token
    if window_k is not None and window_k > 0:
        # With window constraint: check if there's a valid future token within window (vectorized)
        positions_i = torch.arange(seqlen, device=device)[:, None]  # [T, 1]
        positions_j = torch.arange(seqlen, device=device)[None, :]  # [1, T]

        # Check if j is in range (i, i+window_k] for each i
        in_future = positions_j > positions_i  # j > i
        in_window = positions_j <= positions_i + window_k  # j <= i + window_k
        in_range = in_future & in_window  # [T, T]

        # For each position i in each batch, check if any j in range is valid
        future_exists = torch.einsum('bt,st->bs', attention_mask.float(), in_range.float()) > 0  # [B, T]
        future_valid = future_exists & attention_mask.bool()
    else:
        # No window constraint: valid if there's any future token
        token_lens = attention_mask.sum(dim=1)
        max_valid_index = torch.clamp(token_lens - 1, min=0)
        positions_batch = torch.arange(seqlen, device=device).unsqueeze(0)
        future_valid = (positions_batch < max_valid_index.unsqueeze(1)) & attention_mask.bool()

    return future_mask, future_valid


def build_future_mask_from_batch_cu(
    cu_seqlens: torch.Tensor,
    attention_mask: torch.Tensor,
    window_k: int | None,
    dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build document-aware anti-causal attention mask from batch-level cu_seqlens.
    Optimized version with vectorized operations.

    Args:
        cu_seqlens: (B, max_num_docs+1) with -1 padding for invalid entries.
                    Each row contains document boundaries for one sample.
        attention_mask: (B, T) with 1 for valid tokens, 0 for padding.
        window_k: Optional window size for future attention. If None, attend to all future in same doc.
        dtype: dtype to use for the additive mask.

    Returns:
        future_attn_mask: (B, 1, T, T) additive mask, 0 where attending is allowed, large negative elsewhere.
        future_valid: (B, T) bool mask where at least one valid future token exists within same document.
    """
    bsz, seqlen = attention_mask.shape
    device = attention_mask.device
    neg_inf = -1e4

    # Initialize future mask: block everything initially
    future_mask = torch.full(
        (bsz, 1, seqlen, seqlen),
        fill_value=neg_inf,
        device=device,
        dtype=dtype
    )

    # Initialize future_valid: no valid future initially
    future_valid = torch.zeros((bsz, seqlen), dtype=torch.bool, device=device)

    # Process each sample in the batch
    for b in range(bsz):
        cu = cu_seqlens[b]
        # Filter out padding (-1)
        valid_cu = cu[cu >= 0]

        if valid_cu.numel() < 2:
            continue

        # For each document in this sample
        for doc_idx in range(len(valid_cu) - 1):
            doc_start = int(valid_cu[doc_idx].item())
            doc_end = int(valid_cu[doc_idx + 1].item())

            if doc_start >= doc_end:
                continue

            doc_len = doc_end - doc_start

            # Create anti-causal mask for this document (vectorized)
            # Position i attends to position j where j > i within the document
            i_idx = torch.arange(doc_len, device=device)[:, None]  # [doc_len, 1]
            j_idx = torch.arange(doc_len, device=device)[None, :]  # [1, doc_len]

            # Anti-causal: j > i
            doc_mask = (j_idx > i_idx).to(dtype=dtype)  # [doc_len, doc_len]

            # Apply window constraint if specified
            if window_k is not None and window_k > 0:
                distance = j_idx - i_idx
                doc_mask = doc_mask * (distance <= window_k).to(dtype=dtype)

            # Convert to additive mask: 0 where allowed, neg_inf where blocked
            doc_mask = torch.where(doc_mask > 0, 0.0, neg_inf)

            # Place into the full mask
            future_mask[b, 0, doc_start:doc_end, doc_start:doc_end] = doc_mask

            # Mark positions with valid future (vectorized)
            if window_k is not None and window_k > 0:
                # Position i has valid future if min(i + window_k, doc_end - 1) > i
                positions = torch.arange(doc_start, doc_end, device=device)
                future_end = torch.minimum(
                    positions + window_k,
                    torch.tensor(doc_end - 1, device=device, dtype=positions.dtype)
                )
                has_future = future_end > positions
                future_valid[b, doc_start:doc_end] = has_future
            else:
                # All positions except the last have valid future
                future_valid[b, doc_start:doc_end-1] = True

    # Apply padding mask
    pad_mask = (attention_mask == 0).to(dtype=dtype)
    future_mask = future_mask + pad_mask[:, None, None, :] * neg_inf

    return future_mask, future_valid


def _segment_ids_from_cu_seqlens(cu_seqlens: torch.Tensor) -> tuple[torch.Tensor, int]:
    """
    Derive segment ids for each token position from cu_seqlens.
    Returns (segment_id, total_len).
    """
    if cu_seqlens.dim() == 2:
        cu = cu_seqlens[0]
    else:
        cu = cu_seqlens
    total_len = int(cu[-1].item())
    positions = torch.arange(total_len, device=cu.device, dtype=cu.dtype)
    segment_id = torch.bucketize(positions, cu[1:],right=True)
    return segment_id, total_len


def build_future_mask_from_cu(
    cu_seqlens: torch.Tensor, window_k: int | None, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a dense additive mask (B=1) and future_valid from cu_seqlens for an anti-causal (+window) setting.
    Mask shape: [1, 1, T, T], 0 for allowed, -inf otherwise. future_valid shape: [1, T].
    """
    cu = cu_seqlens[0] if cu_seqlens.dim() == 2 else cu_seqlens
    if cu.numel() == 0 or cu[-1] == 0:
        return None, None
    segment_id, total_len = _segment_ids_from_cu_seqlens(cu)
    pos = torch.arange(total_len, device=device, dtype=cu.dtype)
    diff = pos[None, :] - pos[:, None]  # kv_idx - q_idx
    same = segment_id[:, None] == segment_id[None, :]
    future = diff > 0
    if window_k is not None and window_k > 0:
        future = future & (diff <= window_k)
    allowed = same & future
    neg_inf = -torch.finfo(dtype).max
    mask = torch.full((total_len, total_len), fill_value=neg_inf, device=device, dtype=dtype)
    mask = mask.masked_fill(allowed, 0.0).unsqueeze(0).unsqueeze(0)

    seg_end = cu[segment_id + 1]
    future_len = seg_end - pos - 1
    if window_k is not None and window_k > 0:
        future_len = torch.minimum(future_len, torch.tensor(window_k, device=device, dtype=future_len.dtype))
    future_valid = (future_len > 0).unsqueeze(0)
    return mask, future_valid


def _future_valid_from_cu(cu_seqlens: torch.Tensor, window_k: int | None, device: torch.device) -> torch.Tensor | None:
    """Compute future_valid mask [1, T] from cu_seqlens with optional window; return None if empty."""
    cu = cu_seqlens[0] if cu_seqlens.dim() == 2 else cu_seqlens
    if cu.numel() == 0 or cu[-1] == 0:
        return None
    pos = torch.arange(cu[-1], device=device, dtype=cu.dtype)
    seg_end = cu[torch.bucketize(pos, cu[1:], right=True) + 1]
    future_len = seg_end - pos - 1
    if window_k not in (None, 0):
        future_len = torch.minimum(future_len, torch.tensor(window_k, device=device, dtype=cu.dtype))
    return (future_len > 0).unsqueeze(0)


def _register_future_flex_attn():
    """Register a flex attention implementation for future (anti-causal) masks."""
    global _FLEX_REG_DONE
    if _FLEX_REG_DONE or not (_HAS_FLEX_ATTENTION and _HAS_FLEX_INTERFACE):
        return

    def future_flex_attention_forward(module, query, key, value, attention_mask=None, **kwargs):
        # attention_mask here is expected to be a BlockMask
        # Handle grouped-query attention: flex_attention expects the same head count for q and k/v.
        if query.size(1) != key.size(1):
            repeat_factor = query.size(1) // key.size(1)
            key = key.repeat_interleave(repeat_factor, dim=1)
            value = value.repeat_interleave(repeat_factor, dim=1)
        out = flex_attention(query, key, value, block_mask=attention_mask)
        return out, None

    def future_flex_block_mask(
        batch_size: int,
        cache_position,
        kv_length: int,
        kv_offset: int = 0,
        mask_function=None,
        attention_mask=None,
        cu_seqlens=None,
        future_window_k=None,
        **kwargs,
    ):
        # If caller already provided a BlockMask, just reuse it.
        if attention_mask is not None and attention_mask.__class__.__name__ == "BlockMask":
            return attention_mask
        # Otherwise try to build from cu_seqlens if available.
        if cu_seqlens is None:
            # Fallback to causal mask construction from attention_mask if nothing else is available.
            return None
        cu = cu_seqlens[0] if cu_seqlens.dim() == 2 else cu_seqlens
        if cu.numel() == 0 or cu[-1] == 0:
            return None
        device = cache_position.device
        cu = cu.to(device)
        total_len = int(cu[-1].item())
        positions = torch.arange(total_len, device=device, dtype=cu.dtype)
        segment_id = torch.bucketize(positions, cu[1:],right=True)
        config = kwargs.get("config", None)
        cfg_window = getattr(config, "_future_window_k", None) if config is not None else None
        future_window = future_window_k if future_window_k not in (None, 0) else cfg_window

        def mask_mod(_b, _h, q_idx, kv_idx):
            same = segment_id[q_idx] == segment_id[kv_idx]
            future = kv_idx > q_idx
            if future_window is not None:
                future = future & (kv_idx - q_idx <= future_window)
            return same & future

        H = config.num_attention_heads if config is not None else 1
        B = batch_size
        key = (
            device,
            B,
            H,
            total_len,
            future_window,
            tuple(cu.cpu().tolist()),
        )
        block_mask = _BLOCK_MASK_CACHE.get(key)
        if block_mask is None or block_mask.device != device:
            block_mask = create_block_mask(mask_mod, B, H, total_len, total_len, device=device)
            _BLOCK_MASK_CACHE[key] = block_mask
        return block_mask

    AttentionInterface.register("future_flex", future_flex_attention_forward)
    AttentionMaskInterface.register("future_flex", future_flex_block_mask)
    _FLEX_REG_DONE = True


def _parallelize_aux_module(module, world_mesh, parallel_dims, job_config):
    if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)
        apply_fsdp(
            module,
            world_mesh[tuple(dp_mesh_dim_names)],
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=job_config.training.enable_cpu_offload,
            reshard_after_forward_policy=job_config.training.fsdp_reshard_after_forward,
        )
        return True
    if parallel_dims.dp_replicate_enabled:
        if world_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_ddp(
            module,
            world_mesh,
            enable_compile=job_config.training.compile,
            enable_compiled_autograd=job_config.experimental.enable_compiled_autograd,
        )
        return True
    return False

def build_tokenizer(job_config: JobConfig) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(job_config.model.tokenizer_path)


register_train_spec(
    TrainSpec(
        name="fla",
        cls=AutoModelForCausalLM,
        config=AutoConfig,
        parallelize_fn=parallelize_fla,
        pipelining_fn=pipeline_fla,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dataloader,
        build_tokenizer_fn=build_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)


# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(job_config: JobConfig):
    logger.info(f"Starting job: {job_config.job.description}")

    if job_config.experimental.custom_model_path:
        utils.import_module_from_path(job_config.experimental.custom_model_path)

    # used for colorful printing
    color = utils.NoColor if job_config.metrics.disable_color_printing else utils.Color

    if job_config.job.print_args:
        logger.info(
            f"{color.green}{json.dumps(job_config.to_dict(), indent=2, sort_keys=True)}{color.reset}"
        )

    # Enable DFT loss mode and disable auxiliary future/action modules when requested.
    dft_enabled = bool(getattr(job_config, "dft", None) and job_config.dft.enable)
    if dft_enabled:
        if getattr(job_config.future_encoder, "enable", False):
            logger.info("DFT enabled: disabling future_encoder (MI teacher forward).")
            job_config.future_encoder.enable = False
        if getattr(job_config.future_predictor, "enable", False):
            logger.info("DFT enabled: disabling future_predictor head.")
            job_config.future_predictor.enable = False
        if getattr(job_config.action_layer, "enable", False):
            logger.info("DFT enabled: disabling action_layer.")
            job_config.action_layer.enable = False

    # take control of garbage collection to avoid stragglers
    gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

    device_module, device_type = utils.device_module, utils.device_type
    device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
    # Device has to be set before creating TorchFT manager.
    device_module.set_device(device)
    ft_manager = init_ft_manager(job_config)

    # init distributed
    world_size = int(os.environ["WORLD_SIZE"])
    if not ft_manager.enabled:
        parallel_dims = ParallelDims(
            dp_shard=job_config.training.data_parallel_shard_degree,
            dp_replicate=job_config.training.data_parallel_replicate_degree,
            cp=job_config.experimental.context_parallel_degree,
            tp=job_config.training.tensor_parallel_degree,
            pp=job_config.experimental.pipeline_parallel_degree,
            world_size=world_size,
            enable_loss_parallel=not job_config.training.disable_loss_parallel,
        )
    else:
        parallel_dims = FTParallelDims(
            dp_shard=job_config.training.data_parallel_shard_degree,
            dp_replicate=job_config.training.data_parallel_replicate_degree,
            cp=job_config.experimental.context_parallel_degree,
            tp=job_config.training.tensor_parallel_degree,
            pp=job_config.experimental.pipeline_parallel_degree,
            world_size=world_size,
            enable_loss_parallel=not job_config.training.disable_loss_parallel,
            ft_manager=ft_manager,
        )
    dist_utils.init_distributed(job_config)
    # initialize device memory monitor and get peak flops for MFU calculation
    device_memory_monitor = build_device_memory_monitor()
    gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type=device_type)
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    if parallel_dims.pp_enabled:
        raise NotImplementedError(
            "Pipeline parallelism is not supported in this version"
        )
        """
        ! TODO[flame]: We need to fix the pipeline parallelism for flame
        [x] Match the key of models' components with the actual naming
        [ ] Fix the post-init and tie-embedding for pipeline parallelism, HF's transformer automatically
            forces to tie if head is None, we need to handle this case
        [ ]
        """
        pp_mesh = world_mesh["pp"]

    # Set random seed, and maybe enable deterministic mode (mainly for debugging, expect perf loss)
    dist_utils.set_determinism(
        world_mesh, device, job_config.training.seed, job_config.training.deterministic
    )
    train_spec = get_train_spec(job_config.model.name)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        job_config.model.tokenizer_path,
        trust_remote_code=True,
        model_max_length=int(1e10),
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            logger.warning("Tokenizer has no pad token, using eos_token as pad_token.")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            logger.warning("Tokenizer has no pad token or eos token; adding [PAD] token.")
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    logger.info(f"{tokenizer}")
    logger.info(
        f"Loading dataset {job_config.training.dataset}"
        f":{job_config.training.dataset_name}"
        if job_config.training.dataset_name is not None
        else ""
    )
    global_batch_size = (
        job_config.training.batch_size
        * dp_degree
        * job_config.training.gradient_accumulation_steps
    )
    dataset = build_dataset(
        dataset=job_config.training.dataset,
        dataset_name=job_config.training.dataset_name,
        dataset_split=job_config.training.dataset_split,
        data_dir=job_config.training.data_dir,
        data_files=job_config.training.data_files,
        data_probs=job_config.training.data_probs,
        streaming=job_config.training.streaming,
        dp_degree=dp_degree,
        num_workers=job_config.training.num_workers,
        seed=job_config.training.seed,
        trust_remote_code=job_config.training.trust_remote_code,
        seq_len=job_config.training.seq_len,
        eos_token_id=tokenizer.eos_token_id,
    )
    dataset_size = getattr(dataset, "_flame_num_rows", None)
    if job_config.training.epochs is not None:
        if dataset_size is None:
            raise ValueError(
                "training.epochs is set but dataset size is unknown (streaming or interleaved dataset). "
                "Disable streaming and use a single map-style dataset to enable epoch-based scheduling."
            )
        computed_steps = math.ceil(
            dataset_size * job_config.training.epochs / global_batch_size
        )
        logger.info(
            f"Overriding training.steps to {computed_steps} based on epochs={job_config.training.epochs}, "
            f"dataset_size={dataset_size}, global_batch_size={global_batch_size}"
        )
        job_config.training.steps = computed_steps
    if dp_rank == 0:
        _log_sample_preview(dataset, tokenizer, color)

    logger.info("Building dataloader...")
    dataloader = build_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        rank=dp_rank,
        world_size=dp_degree,
        batch_size=job_config.training.batch_size,
        seq_len=job_config.training.seq_len,
        context_len=job_config.training.context_len,
        varlen=job_config.training.varlen,
        respect_doc_boundaries=job_config.future_encoder.respect_doc_boundaries,
        num_workers=job_config.training.num_workers,
        pin_memory=job_config.training.pin_memory,
        persistent_workers=job_config.training.persistent_workers,
        snapshot_every_n_steps=job_config.checkpoint.interval,
        sample_level=job_config.training.sample_level,
    )

    logger.info(f"Loading model config from {job_config.model.config}")
    model_config = AutoConfig.from_pretrained(job_config.model.config)
    if getattr(model_config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model_config.pad_token_id = tokenizer.pad_token_id
    # set the model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. disable fused norm if TP is enabled
    # 3. vocab size from tokenizer
    # 4. context_len base on inputs
    if parallel_dims.tp_enabled:
        if model_config.fuse_norm:
            logger.warning(
                f"{color.red}"
                f"Fused norm is not compatible with tensor parallelism. "
                f"Disabling it for now."
                f"{color.reset}"
            )
            model_config.fuse_norm = False
    if parallel_dims.loss_parallel_enabled:
        if model_config.fuse_linear_cross_entropy:
            logger.warning(
                f"{color.red}"
                f"Loss parallel enabled. Disabling fused cross entropy for now."
                f"{color.reset}"
            )
            model_config.fuse_linear_cross_entropy = False
    if dft_enabled and getattr(model_config, "fuse_linear_cross_entropy", False):
        logger.info("DFT enabled: disabling fused linear cross entropy kernel.")
        model_config.fuse_linear_cross_entropy = False
    model_config.vocab_size = max(tokenizer.vocab_size, model_config.vocab_size)

    logger.info(
        f"Building model from the config\n{color.green}{model_config}{color.reset}"
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(model_config)
        if (
            getattr(model_config, "fuse_linear_cross_entropy", False)
            and FusedLinearCrossEntropyLoss is not None
        ):
            model.criterion = FusedLinearCrossEntropyLoss(
                num_chunks=8 // parallel_dims.tp
            )
        # defer weight initialization until after parallelisms are applied
        model.apply(lambda m: setattr(m, "_is_hf_initialized", False))
    logger.info(f"{color.blue}\n{model}{color.reset}\n")

    # Build the collection of model converters. No-op if `model.converters` empty
    model_converters = build_model_converters(job_config, parallel_dims)
    model_converters.convert(model)
    global _MODEL_CONVERTERS
    _MODEL_CONVERTERS = model_converters

    # calculate model size and flops per token
    model_param_count, num_flops_per_token = get_nparams_and_flops(
        model, model_config, job_config.training.context_len
    )

    # move sharded model to CPU/GPU and initialize weights via DTensor
    if job_config.checkpoint.create_seed_checkpoint:
        init_device = "cpu"
    elif job_config.training.enable_cpu_offload:
        init_device = "cpu"
    else:
        init_device = device_type

    # apply parallelisms and initialization
    if parallel_dims.pp_enabled:
        # apply PT-D Pipeline Parallel
        (
            pp_schedule,
            model_parts,
            has_first_stage,
            has_last_stage,
        ) = train_spec.pipelining_fn(
            model,
            pp_mesh,
            parallel_dims,
            job_config,
            device,
            model_config,
            train_spec.loss_fn,
        )
        # when PP is enabled, `model` obj is no longer used after this point, model_parts is used instead
        del model

        # For PP with looped schedules, each item in model_parts is one stage-model-chunk.
        # We need to iterate through model_parts to apply SPMD parallelisms, compilation,
        # optimizer, and checkpointing
        for m in model_parts:
            # apply SPMD-style PT-D techniques
            train_spec.parallelize_fn(m, world_mesh, parallel_dims, job_config)
            m.to_empty(device=init_device)
            with torch.no_grad():
                m.post_init()
                # 关键修复：手动初始化 RoPE inv_freq buffer
                # HF 模型将 inv_freq 注册为 non-persistent buffer，不会保存到 checkpoint
                init_rope_inv_freq(m, init_device)
            m.train()

        # confirm that user will be able to view loss metrics on the console
        ensure_pp_loss_visible(parallel_dims, job_config, color)
    else:
        # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
        train_spec.parallelize_fn(model, world_mesh, parallel_dims, job_config)
        model.to_empty(device=init_device)
        with torch.no_grad():
            model.post_init()
            # 关键修复：手动初始化 RoPE inv_freq buffer
            # HF 模型将 inv_freq 注册为 non-persistent buffer，不会保存到 checkpoint
            init_rope_inv_freq(model, init_device)
        model.train()

        model_parts = [model]

    # Initialize Future predictor and MI Estimator if enabled
    mi_estimator = None
    future_predictor = None
    action_layer = None
    if job_config.future_encoder.enable:
        logger.info("Initializing Future Predictor and MI Estimator...")
        if job_config.future_predictor.enable:
            future_predictor = FuturePredictorHead(
                hidden_size=model_config.hidden_size,
                head_type=job_config.future_predictor.head_type,
                dropout=job_config.future_predictor.dropout,
            )
            future_predictor.to(init_device)
            future_predictor.train()
            if len(list(future_predictor.parameters())) > 0:
                _parallelize_aux_module(
                    future_predictor, world_mesh, parallel_dims, job_config
                )
                model_parts.append(future_predictor)

        mi_estimator = build_mi_estimator(
            estimator_type=job_config.future_encoder.estimator_type,
            hidden_size=model_config.hidden_size,
            temperature=job_config.future_encoder.temperature,
        )
        mi_estimator.to(init_device)
        mi_estimator.train()
        if len(list(mi_estimator.parameters())) > 0:
            _parallelize_aux_module(mi_estimator, world_mesh, parallel_dims, job_config)
            model_parts.append(mi_estimator)

    if job_config.action_layer.enable:
        if not job_config.future_encoder.enable:
            raise ValueError("Action layer requires future_encoder.enable for future summaries.")
        use_rms_norm = job_config.action_layer.use_rms_norm
        if not use_rms_norm:
            use_rms_norm = bool(
                getattr(model_config, "norm_type", "").lower() == "rmsnorm"
                or hasattr(model_config, "rms_norm_eps")
            )
        ffn_hidden_size = job_config.action_layer.ffn_hidden_size
        if ffn_hidden_size is None:
            ffn_hidden_size = getattr(model_config, "intermediate_size", None)
        activation = job_config.action_layer.activation
        if activation is None:
            activation = getattr(model_config, "hidden_act", None)
        action_layer = ActionLayer(
            hidden_size=model_config.hidden_size,
            top_k=job_config.action_layer.top_k,
            head_type=job_config.action_layer.head_type,
            tau=job_config.action_layer.tau,
            reward_clamp=job_config.action_layer.reward_clamp,
            mean_threshold=job_config.action_layer.mean_threshold,
            gt_bias=job_config.action_layer.gt_bias,
            use_rms_norm=use_rms_norm,
            ffn_hidden_size=ffn_hidden_size,
            activation=activation,
            residual=job_config.action_layer.residual,
            delta_init_zero=job_config.action_layer.delta_init_zero,
            score_type=job_config.action_layer.score_type,
        )
        action_layer.to(init_device)
        action_layer.train()
        if len(list(action_layer.parameters())) > 0:
            _parallelize_aux_module(action_layer, world_mesh, parallel_dims, job_config)
            model_parts.append(action_layer)

    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )

    freeze_lm_for_infonce = getattr(job_config.experimental, "freeze_lm_for_infonce", False)
    if freeze_lm_for_infonce:
        logger.info("Freezing LM parameters and skipping CE loss (InfoNCE-only sanity check).")
        for p in model_parts[0].parameters():
            p.requires_grad = False

    # build optimizer after applying parallelisms to the model
    optim_model_parts = (
        [m for m in model_parts if any(p.requires_grad for p in m.parameters(recurse=True))]
        if freeze_lm_for_infonce
        else model_parts
    )
    optimizers = train_spec.build_optimizers_fn(optim_model_parts, job_config, ft_manager)

    # Optionally scale LR / override weight decay for the future predictor only.
    fp_lr_scale = getattr(job_config.future_predictor, "lr_scale", 1.0)
    fp_wd_override = getattr(job_config.future_predictor, "weight_decay", None)
    fp_idx = None
    if future_predictor is not None and hasattr(optimizers, "optimizers"):
        try:
            fp_idx = model_parts.index(future_predictor)
            fp_optim = optimizers.optimizers[fp_idx]
            for pg in fp_optim.param_groups:
                if fp_lr_scale != 1.0:
                    pg["lr"] = pg["lr"] * fp_lr_scale
                if fp_wd_override is not None:
                    pg["weight_decay"] = fp_wd_override
            logger.info(
                f"Applied lr_scale={fp_lr_scale} "
                f"{'and weight_decay='+str(fp_wd_override) if fp_wd_override is not None else ''} "
                "to future_predictor optimizer"
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(f"Failed to apply lr/wd override to future_predictor: {exc}")

    lr_schedulers = train_spec.build_lr_schedulers_fn(optimizers, job_config)
    # Post optimizer step model converters hook.
    # e.g. calculate float8 dynamic amax/scale for all-parameter for FSDP2
    # where it issues a single all-reduce for all parameters at once for better performance
    global _MODEL_CONVERTER_HOOK_MODEL_PARTS
    _MODEL_CONVERTER_HOOK_MODEL_PARTS = model_parts
    optimizers.register_step_post_hook(_optimizer_post_step_hook)

    train_state = TrainState()

    # load initial checkpoint
    checkpoint = CheckpointManager(
        dataloader=dataloader,
        model_parts=model_parts,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        states={"train_state": train_state},
        job_config=job_config,
        ft_manager=ft_manager,
    )
    
    _ensure_cloudpickle_for_dist_objects()

    if job_config.checkpoint.create_seed_checkpoint:
        assert world_size == 1, (
            "Must create seed checkpoint using a single device, to disable sharding"
        )
        assert job_config.checkpoint.enable_checkpoint, (
            "Must enable checkpointing when creating a seed checkpoint"
        )
        checkpoint.save(curr_step=0, force=True)
        logger.info("Created seed checkpoint")
        return

    checkpoint.load(step=job_config.checkpoint.load_step)
    metric_logger = build_metrics_processor(job_config, parallel_dims)
    # Set dependent attributes for metric_logger
    metric_logger.num_flops_per_token = num_flops_per_token
    metric_logger.optimizers = optimizers  # Pass optimizers if needed by logger logic
    metric_logger.lr_schedulers = (
        lr_schedulers  # Pass schedulers if needed by logger logic
    )

    # plot losses loaded from checkpoint (if any) to TensorBoard
    # NOTE: Loss info after the last log step before checkpoint saving will not be ploted.
    #       This can be avoided by setting checkpoint.interval to be a multiple of metrics.log_freq
    if train_state.step > 0 and len(metric_logger.data_loading_times) > 0:
        for idx, step in enumerate(train_state.log_steps):
            metric_logger.log(
                step,
                global_avg_loss=train_state.global_avg_losses[idx],
                global_max_loss=train_state.global_max_losses[idx],
            )

    data_iterator = iter(dataloader)

    train_context = dist_utils.get_train_context(
        parallel_dims.loss_parallel_enabled,
        job_config.experimental.enable_compiled_autograd,
    )
    maybe_enable_amp = dist_utils.maybe_enable_amp(
        parallel_dims,
        job_config.training.mixed_precision_param,
        device_type,
    )

    # variables used to keep info for metrics logging
    device_memory_monitor.reset_peak_stats()

    num_tokens_per_step = global_batch_size * job_config.training.seq_len
    # train loop
    logger.info(f"{color.red}***** Running training *****{color.reset}")
    logger.info(f"{color.green}  Training starts at step {train_state.step + 1}")
    logger.info(
        f"{color.green}  Number of tokens per sequence = {job_config.training.seq_len:,}"
    )
    logger.info(
        f"{color.green}  Gradient Accumulation steps = {job_config.training.gradient_accumulation_steps}"
    )
    logger.info(
        f"{color.green}  Instantaneous batch size (per device) = {job_config.training.batch_size:,}"
    )
    logger.info(
        f"{color.green}  Global batch size (w. parallel, distributed & accumulation) = {global_batch_size:,}"
        f" ({num_tokens_per_step:,} tokens)"
    )
    steps_per_epoch = None
    tokens_per_epoch = None
    if job_config.training.epochs is not None and dataset_size is not None:
        steps_per_epoch = math.ceil(dataset_size / global_batch_size)
        tokens_per_epoch = steps_per_epoch * num_tokens_per_step
        logger.info(
            f"{color.green}  Epoch mode: epochs = {job_config.training.epochs:,}, "
            f"steps/epoch ≈ {steps_per_epoch:,} ({tokens_per_epoch:,} tokens), "
            f"dataset_size = {dataset_size:,}{color.reset}"
        )
    logger.info(
        f"{color.green}  Total optimization steps = {job_config.training.steps:,} "
        f"({job_config.training.steps * num_tokens_per_step:,} tokens)"
    )
    logger.info(
        f"{color.green}  Warmup steps = {job_config.lr_scheduler.warmup_steps:,}"
        f" ({job_config.lr_scheduler.warmup_steps * num_tokens_per_step:,} tokens)"
    )
    logger.info(
        f"{color.green}  Number of parameters = {model_param_count:,} {color.reset}"
    )
    dft_mix_ce_ratio = getattr(job_config.dft, "mix_ce_ratio", 0.0) if dft_enabled else 0.0

    with (
        maybe_enable_profiling(
            job_config, global_step=train_state.step
        ) as torch_profiler,
        maybe_enable_memory_snapshot(
            job_config, global_step=train_state.step
        ) as memory_profiler,
    ):
        while train_state.step < job_config.training.steps:
            train_state.step += 1
            gc_handler.run(train_state.step)

            optimizers.zero_grad()

            # Apply future predictor warmup scaling (on top of lr_scale).
            fp_warmup_steps = getattr(job_config.future_predictor, "warmup_steps", 0)
            fp_warmup_factor = 1.0
            if (
                fp_idx is not None
                and fp_warmup_steps > 0
                and hasattr(optimizers, "optimizers")
            ):
                fp_warmup_factor = min(1.0, train_state.step / fp_warmup_steps)
                try:
                    fp_optim = optimizers.optimizers[fp_idx]
                    for pg in fp_optim.param_groups:
                        base_lr = pg.get("fp_base_lr", pg["lr"])
                        pg["lr"] = base_lr * fp_warmup_factor
                except Exception as exc:  # pragma: no cover - defensive fallback
                    logger.warning(f"Failed to apply future predictor warmup lr: {exc}")

            inv_grad_acc_steps = 1.0 / job_config.training.gradient_accumulation_steps
            losses = []
            ce_losses = []
            ce_unweighted_losses = []
            target_prob_means = []
            aux_losses = []
            action_losses = []
            action_metrics_acc = {}
            action_metrics_count = 0
            # mi_lower_bounds = []
            # counts = []
            # do gradient accumulation if enabled
            for _ in range(job_config.training.gradient_accumulation_steps):
                # get batch
                data_load_start = time.perf_counter()
                batch = next(data_iterator)
                input_ids, labels = batch["input_ids"], batch["labels"]

                # Update metrics processor state before forward/backward
                metric_logger.ntokens_since_last_log += labels.numel()
                metric_logger.data_loading_times.append(
                    time.perf_counter() - data_load_start
                )

                input_ids = input_ids.to(device_type)

                """
                TODO[flame]: We need to carefully handle the position_ids for TP/CP
                Depending on the Models'PE, the position_ids might be different.

                e.g. for TP
                    For RoPE, all ranks have the same position_ids. [FOR HF model]
                    For sinusoidal, each rank has the coresponding chunked  position_ids. [FOR HF model]

                e.g. for CP, [optional_context_parallel_ctx shoudl automatically distbute the position_ids]
                    Each rank has the coresponding chunked position_ids. [FOR All model]

                """
                labels = labels.to(device_type)
                attention_mask = (
                    batch["attention_mask"].to(device_type)
                    if "attention_mask" in batch
                    else None
                )
                if attention_mask is not None and attention_mask.dtype not in (
                    torch.bool,
                    torch.float16,
                    torch.float32,
                    torch.bfloat16,
                ):
                    attention_mask = attention_mask.to(dtype=torch.bool)
                cu_seqlens = (
                    batch["cu_seqlens"].to(device_type)
                    if "cu_seqlens" in batch and batch["cu_seqlens"] is not None
                    else None
                )
                # Handle position_ids based on cu_seqlens format
                if cu_seqlens is not None and cu_seqlens.dim() == 1:
                    # Varlen mode: cu_seqlens is 1D, use prepare_position_ids
                    position_ids = prepare_position_ids(cu_seqlens).to(torch.int32)
                else:
                    # Non-varlen mode (batch-level cu_seqlens is 2D) or no cu_seqlens
                    # Use standard sequential position_ids for each batch element
                    position_ids = (
                        torch.arange(0, input_ids.shape[1], device=device_type)
                        .repeat(input_ids.shape[0], 1)
                        .to(torch.int32)
                    )
                if position_ids.dim() == 1:
                    position_ids = position_ids.unsqueeze(0)
                # apply context parallelism if cp is enabled
                # ensure CP handles the separate freqs_cis buffer for each pp stage
                optional_context_parallel_ctx = (
                    dist_utils.create_context_parallel_ctx(
                        cp_mesh=world_mesh["cp"],
                        cp_buffers=[input_ids, labels, position_ids],
                        cp_seq_dims=[1, 1, 1],
                        cp_no_restore_buffers={input_ids, labels, position_ids},
                        cp_rotate_method=job_config.experimental.context_parallel_rotate_method,
                    )
                    if parallel_dims.cp_enabled
                    else None
                )

                # #! TODO[flame], we should distribute the position_ids as well with CP
                if parallel_dims.pp_enabled:
                    raise NotImplementedError(
                        "Pipeline parallelism is not supported in this version"
                    )
                    # Pipeline Parallel forward / backward inside step() call
                    with train_context(optional_context_parallel_ctx):
                        targets, losses = (
                            (labels, []) if has_last_stage else (None, None)
                        )

                        if has_first_stage:
                            pp_schedule.step(input_ids, target=targets, losses=losses)
                        else:
                            pp_schedule.step(target=targets, losses=losses)

                    # accumulate losses across pipeline microbatches
                    # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
                    loss = (
                        torch.mean(torch.stack(losses)).to(device)
                        if has_last_stage
                        else torch.tensor([-1.0], device=device)
                    )
                else:
                    # Non-PP forward / backward
                    with train_context(optional_context_parallel_ctx):
                        lm_forward_ctx = torch.no_grad() if freeze_lm_for_infonce else nullcontext()
                        forward_kwargs = dict(
                            input_ids=input_ids,
                            position_ids=position_ids,
                            attention_mask=attention_mask,
                            cu_seqlens=cu_seqlens,
                            output_hidden_states=job_config.future_encoder.enable,
                            use_cache=False,
                        )
                        if not freeze_lm_for_infonce and not dft_enabled:
                            forward_kwargs["labels"] = labels
                        with lm_forward_ctx, maybe_enable_amp:
                            output = model(**forward_kwargs)
                            logits = output.logits
                            target_prob_mean = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
                            if dft_enabled:
                                shift_logits = logits[:, :-1, :].contiguous()
                                shift_labels = labels[:, 1:].contiguous()
                                vocab_size = shift_logits.size(-1)
                                shift_logits_flat = shift_logits.view(-1, vocab_size)
                                shift_labels_flat = shift_labels.view(-1)
                                valid_mask = shift_labels_flat != -100
                                if attention_mask is not None:
                                    valid_mask = valid_mask & attention_mask[:, 1:].reshape(-1).to(torch.bool)
                                if valid_mask.any():
                                    selected_logits = shift_logits_flat[valid_mask]
                                    selected_labels = shift_labels_flat[valid_mask]
                                    ce_tokens = F.cross_entropy(
                                        selected_logits,
                                        selected_labels,
                                        reduction="none",
                                    )
                                    probs = torch.softmax(selected_logits, dim=-1)
                                    target_probs = probs.gather(1, selected_labels.unsqueeze(-1)).squeeze(-1).detach()
                                    target_prob_mean = target_probs.mean()
                                    dft_tokens = ce_tokens * target_probs
                                    dft_mean = dft_tokens.mean()
                                    ce_unweighted_mean = ce_tokens.mean()
                                else:
                                    zero = shift_logits_flat.new_zeros(())
                                    dft_mean = zero
                                    ce_unweighted_mean = zero
                                    target_prob_mean = zero
                                ce_loss_raw = (
                                    dft_mean
                                    if dft_mix_ce_ratio <= 0.0
                                    else dft_mix_ce_ratio * ce_unweighted_mean + (1.0 - dft_mix_ce_ratio) * dft_mean
                                )
                            else:
                                if freeze_lm_for_infonce:
                                    ce_loss_raw = torch.tensor(
                                        0.0, device=device, dtype=output.logits.dtype
                                    )
                                    ce_unweighted_mean = ce_loss_raw
                                else:
                                    ce_loss_raw = output.loss
                                    ce_unweighted_mean = ce_loss_raw
                            ce_loss = ce_loss_raw * inv_grad_acc_steps
                            ce_unweighted_scaled = ce_unweighted_mean * inv_grad_acc_steps
                        # If action layer is enabled and ce_loss_weight is set (default 0), scale CE accordingly
                        ce_loss_weight = (
                            job_config.action_layer.ce_loss_weight
                            if job_config.action_layer.enable and not freeze_lm_for_infonce
                            else (0.0 if freeze_lm_for_infonce else 1.0)
                        )
                        scaled_ce = ce_loss * ce_loss_weight
                        total_loss = scaled_ce

                        aux_loss = torch.tensor(0.0, device=device)
                        aux_loss_scaled = torch.tensor(0.0, device=device)
                        action_loss_scaled = torch.tensor(0.0, device=device)
                        action_metrics_step = None
                        future_summaries = None
                        future_valid = None
                        future_summaries_detached = None

                        if job_config.future_encoder.enable:
                            future_summaries_detached = None
                            future_valid = None
                            hidden_states = output.hidden_states[-1]
                            teacher_dtype = hidden_states.dtype
                            # ensure position_ids are 2D for HF attention path
                            future_position_ids = position_ids
                            if future_position_ids is not None and future_position_ids.dim() == 1:
                                future_position_ids = future_position_ids.unsqueeze(0)
                            # Run a second forward (no grad) with anti-causal (+window) mask to get teacher summaries.
                            future_window_k = job_config.future_encoder.future_k
                            if _FLEX_DEBUG and cu_seqlens is not None and _HAS_FLEX_ATTENTION and _HAS_FLEX_INTERFACE:
                                logger.info(
                                    f"[flex_debug] step?_future pass varlen len={int(cu_seqlens.view(-1)[-1])} window={future_window_k}"
                                )
                            if cu_seqlens is not None and _HAS_FLEX_ATTENTION and _HAS_FLEX_INTERFACE:
                                _register_future_flex_attn()
                                prev_impl = getattr(model.config, "_attn_implementation", None)
                                prev_window = getattr(model.config, "_future_window_k", None)
                                model.config._attn_implementation = "future_flex"
                                model.config._future_window_k = future_window_k if future_window_k != 0 else None
                                if hasattr(model, "model") and hasattr(model.model, "config"):
                                    model.model.config._attn_implementation = model.config._attn_implementation
                                    model.model.config._future_window_k = model.config._future_window_k
                                future_forward_kwargs = dict(
                                    input_ids=input_ids,
                                    position_ids=future_position_ids,
                                    cu_seqlens=cu_seqlens,
                                    attention_mask=None,
                                    output_hidden_states=True,
                                    use_cache=False,
                                )
                                if _FLEX_DEBUG:
                                    logger.info(
                                        f"[flex_debug] future flex forward using impl=future_flex window={model.config._future_window_k} "
                                        f"cu_seqlens_len={int(cu_seqlens.view(-1)[-1])} pos_shape={tuple(future_position_ids.shape)}"
                                    )
                                with torch.no_grad():
                                    future_output = model(**future_forward_kwargs)
                                    future_summaries_detached = future_output.hidden_states[-1].detach()
                                model.config._attn_implementation = prev_impl
                                model.config._future_window_k = prev_window
                                if hasattr(model, "model") and hasattr(model.model, "config"):
                                    model.model.config._attn_implementation = prev_impl
                                    model.model.config._future_window_k = prev_window
                                # Build validity mask: length of allowed future tokens per position
                                future_valid = _future_valid_from_cu(
                                    cu_seqlens,
                                    window_k=future_window_k if future_window_k != 0 else None,
                                    device=input_ids.device,
                                )
                                if future_valid is None:
                                    future_valid = torch.zeros(
                                        (hidden_states.size(0), hidden_states.size(1)),
                                        dtype=torch.bool,
                                        device=hidden_states.device,
                                    )
                            elif cu_seqlens is not None and cu_seqlens.dim() == 1:
                                # Fallback dense future mask for varlen
                                future_mask, future_valid = build_future_mask_from_cu(
                                    cu_seqlens,
                                    window_k=future_window_k if future_window_k != 0 else None,
                                    dtype=teacher_dtype,
                                    device=input_ids.device,
                                )
                                if future_mask is None or future_valid is None:
                                    future_valid = torch.zeros(
                                        (hidden_states.size(0), hidden_states.size(1)),
                                        dtype=torch.bool,
                                        device=hidden_states.device,
                                    )
                                    future_summaries_detached = torch.zeros_like(hidden_states)
                                else:
                                    future_forward_kwargs = dict(
                                        input_ids=input_ids,
                                        position_ids=future_position_ids,
                                        attention_mask=future_mask,
                                        output_hidden_states=True,
                                        use_cache=False,
                                    )
                                    if _FLEX_DEBUG:
                                        logger.info(
                                            f"[flex_debug] future dense forward window={future_window_k} mask_shape={tuple(future_mask.shape)} "
                                            f"cu_seqlens_len={int(cu_seqlens.view(-1)[-1])}"
                                        )
                                    with torch.no_grad():
                                        future_output = model(**future_forward_kwargs)
                                        future_summaries_detached = future_output.hidden_states[-1].detach()
                            elif attention_mask is not None:
                                # Non-varlen path: check if we should use document-aware mask
                                respect_doc_boundaries = job_config.future_encoder.respect_doc_boundaries
                                if respect_doc_boundaries and cu_seqlens is not None and cu_seqlens.dim() == 2:
                                    # Use document-aware future mask based on EOS tokens
                                    future_attn_mask, future_valid = build_future_mask_from_batch_cu(
                                        cu_seqlens=cu_seqlens,
                                        attention_mask=attention_mask,
                                        window_k=future_window_k if future_window_k != 0 else None,
                                        dtype=torch.float32
                                    )
                                else:
                                    # Fallback: simple future mask without document boundaries
                                    future_attn_mask, future_valid = build_future_attention_mask(
                                        attention_mask=attention_mask,
                                        dtype=torch.float32,
                                        window_k=future_window_k if future_window_k != 0 else None
                                    )
                                with torch.no_grad():
                                    future_output = model(
                                        input_ids=input_ids,
                                        position_ids=position_ids,
                                        attention_mask=future_attn_mask,
                                        output_hidden_states=True,
                                        use_cache=False,
                                    )
                                    future_summaries_detached = future_output.hidden_states[-1].detach()
                            if future_valid is not None and future_predictor is not None and mi_estimator is not None:
                                hidden_states = output.hidden_states[-1]
                                # Align targets to future positions (t -> t+k) to avoid using the current token's residual.
                                shift_k = max(1, getattr(job_config.future_encoder, "shift_k", 1))
                                if shift_k > 1:
                                    # Sample a random shift in [1, shift_k] each step to increase difficulty/diversity.
                                    shift = int(torch.randint(1, shift_k + 1, (1,), device=hidden_states.device).item())
                                else:
                                    shift = 1
                                if hidden_states.size(1) > shift:
                                    predicted_future = future_predictor(hidden_states[:, :-shift, :])
                                    future_target = future_summaries_detached[:, shift:, :]
                                    future_valid_shifted = future_valid[:, :-shift] if future_valid is not None else None
                                    aux_loss = mi_estimator(
                                        predicted_future,
                                        future_target,
                                        valid_mask=future_valid_shifted,
                                    )
                                    aux_loss_scaled = (
                                        aux_loss
                                        * inv_grad_acc_steps
                                        * job_config.future_encoder.loss_weight
                                    )
                                else:
                                    aux_loss = torch.tensor(0.0, device=device)
                                    aux_loss_scaled = torch.tensor(0.0, device=device)
                                if metric_logger.should_log(train_state.step) and dist.get_rank() == 0:
                                    valid_tokens = future_valid.sum().item() if future_valid is not None else 0
                                    logger.info(
                                        f"raw_aux={aux_loss.item():.4f}, scaled={aux_loss_scaled.item():.4f}, "
                                        f"valid_tokens={valid_tokens}"
                                        f' log(valid_count) * (1/grad_acc_steps) * loss_weight={math.log(valid_tokens) * inv_grad_acc_steps * job_config.future_encoder.loss_weight if valid_tokens > 0 else 0.0}'
                                    )
                        total_loss = total_loss + aux_loss_scaled

                        if action_layer is not None:
                            if future_summaries_detached is None:
                                raise ValueError("Future summaries are required for the action layer but were not computed.")
                            rl_loss, action_metrics_step = action_layer(
                                logits=output.logits,
                                hidden_states=hidden_states,
                                labels=labels,
                                future_summaries=future_summaries_detached,
                                future_valid=future_valid,
                                embed_weight=model.get_input_embeddings().weight,
                                attention_mask=attention_mask,
                            )
                            action_loss_scaled = (
                                rl_loss
                                * inv_grad_acc_steps
                                * job_config.action_layer.loss_weight
                            )
                            total_loss = total_loss + action_loss_scaled

                        total_loss.backward()

                losses.append(total_loss)
                ce_losses.append(ce_loss.detach())
                ce_unweighted_losses.append(ce_unweighted_scaled.detach())
                if dft_enabled:
                    target_prob_means.append(target_prob_mean.detach())
                aux_losses.append(aux_loss_scaled.detach())
                action_losses.append(action_loss_scaled.detach())
                if action_metrics_step is not None:
                    for k, v in action_metrics_step.items():
                        action_metrics_acc[k] = action_metrics_acc.get(k, 0.0) + v.detach()
                    action_metrics_count += 1
                # mi_lower_bounds.append(mi_lower_bound.detach())
                # counts.append(count.detach())

            loss = sum(losses)
            ce_loss_total = sum(ce_losses)
            ce_unweighted_total = sum(ce_unweighted_losses)
            aux_loss_total = sum(aux_losses)
            action_loss_total = sum(action_losses)
            target_prob_avg = (
                sum(target_prob_means) / len(target_prob_means)
                if target_prob_means
                else torch.tensor(0.0, device=device)
            )
            if action_metrics_count > 0:
                action_metrics_avg = {
                    k: v / action_metrics_count for k, v in action_metrics_acc.items()
                }
            else:
                action_metrics_avg = {
                    k: torch.tensor(0.0, device=device)
                    for k in ["avg_reward", "max_reward", "avg_action_size"]
                }
            # mi_lower_bound_total = sum(mi_lower_bounds)
            # count_total = sum(counts)

            # clip gradients
            grad_norm = dist_utils.clip_grad_norm_(
                [p for m in model_parts for p in m.parameters()],
                job_config.training.max_norm,
                foreach=True,
                pp_mesh=pp_mesh if parallel_dims.pp_enabled else None,
            )

            # optimizer step
            checkpoint.maybe_wait_for_staging()
            if job_config.training.skip_nan_inf and (
                grad_norm.isnan() or grad_norm.isinf()
            ):
                logger.warning(
                    f"Skipping optimizer step - detected invalid gradient norm: {grad_norm:.4f}"
                )
                optimizers.zero_grad()
                train_state.skipped_step += 1
            else:
                optimizers.step()
            lr_schedulers.step()
            # Store base lr for fp after scheduler step so next iteration warmup scales from scheduler value.
            if fp_idx is not None and hasattr(optimizers, "optimizers"):
                try:
                    fp_optim = optimizers.optimizers[fp_idx]
                    for pg in fp_optim.param_groups:
                        pg["fp_base_lr"] = pg["lr"]
                except Exception:
                    pass

            # log metrics - Use MetricsProcessor
            if metric_logger.should_log(train_state.step):
                if (
                    parallel_dims.dp_replicate_enabled
                    or parallel_dims.dp_shard_enabled
                    or parallel_dims.cp_enabled
                ):
                    loss = loss.detach()
                    ce_loss_total = ce_loss_total.detach()
                    ce_unweighted_total = ce_unweighted_total.detach()
                    aux_loss_total = aux_loss_total.detach()
                    action_loss_total = action_loss_total.detach()
                    target_prob_avg = target_prob_avg.detach()
                    # mi_lower_bound_total = mi_lower_bound_total.detach()
                    # Use dist_mean/max on the accumulated loss for the step
                    global_avg_loss, global_max_loss = (
                        dist_utils.dist_mean(
                            loss,
                            world_mesh["dp_cp"],
                        ),
                        dist_utils.dist_max(
                            loss,
                            world_mesh["dp_cp"],
                        ),
                    )
                    global_avg_ce_loss = dist_utils.dist_mean(
                        ce_loss_total, world_mesh["dp_cp"]
                    )
                    global_avg_ce_unweighted_loss = dist_utils.dist_mean(
                        ce_unweighted_total, world_mesh["dp_cp"]
                    )
                    if dft_enabled:
                        global_avg_target_prob = dist_utils.dist_mean(
                            target_prob_avg, world_mesh["dp_cp"]
                        )
                    else:
                        global_avg_target_prob = 0.0
                    
                    if job_config.future_encoder.enable:                       
                        # count = count_total.detach()
                        # global_count = dist_utils.dist_sum(count, world_mesh["dp_cp"])
                        global_avg_aux_loss = dist_utils.dist_mean(aux_loss_total, world_mesh["dp_cp"])
                        # global_avg_mi_lb = dist_utils.dist_mean(mi_lower_bound_total, world_mesh["dp_cp"])
                    else:
                        global_avg_aux_loss = 0.0
                        global_avg_mi_lb = 0.0
                    if job_config.action_layer.enable:
                        global_avg_action_loss = dist_utils.dist_mean(action_loss_total, world_mesh["dp_cp"])
                        global_avg_action_reward = dist_utils.dist_mean(
                            action_metrics_avg["avg_reward"].detach(), world_mesh["dp_cp"]
                        )
                        global_avg_action_size = dist_utils.dist_mean(
                            action_metrics_avg["avg_action_size"].detach(), world_mesh["dp_cp"]
                        )
                        global_max_action_reward = dist_utils.dist_max(
                            action_metrics_avg["max_reward"].detach(), world_mesh["dp_cp"]
                        )
                    else:
                        global_avg_action_loss = 0.0
                        global_avg_action_reward = 0.0
                        global_avg_action_size = 0.0
                        global_max_action_reward = 0.0
                    
                else:
                    # Scale back the loss before logging
                    global_avg_loss = global_max_loss = loss.item()
                    global_avg_ce_loss = ce_loss_total.item()
                    global_avg_ce_unweighted_loss = ce_unweighted_total.item()
                    global_avg_target_prob = target_prob_avg.item() if dft_enabled else 0.0
                    if job_config.future_encoder.enable:
                        global_avg_aux_loss = aux_loss_total.item()
                        # global_avg_mi_lb = mi_lower_bound_total.item()
                    else:
                        global_avg_aux_loss = 0.0
                        # global_avg_mi_lb = 0.0
                    if job_config.action_layer.enable:
                        global_avg_action_loss = action_loss_total.item()
                        global_avg_action_reward = action_metrics_avg["avg_reward"].item()
                        global_avg_action_size = action_metrics_avg["avg_action_size"].item()
                        global_max_action_reward = action_metrics_avg["max_reward"].item()
                    else:
                        global_avg_action_loss = 0.0
                        global_avg_action_reward = 0.0
                        global_avg_action_size = 0.0
                        global_max_action_reward = 0.0

                # Update train state tokens and elapsed time
                time_now = time.perf_counter()
                time_delta = (
                    time_now - metric_logger.time_last_log
                )  # Use metric_logger's time
                train_state.token += (
                    metric_logger.ntokens_since_last_log  # Use tokens tracked by metric_logger
                    * parallel_dims.world_size
                    / parallel_dims.non_data_parallel_size
                )
                train_state.elapsed += timedelta(seconds=time_delta)
                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)

                # Log using the metric processor
                last_lr = lr_schedulers.schedulers[0].get_last_lr()[0]
                eta = (
                    train_state.elapsed
                    * (job_config.training.steps - train_state.step)
                    / train_state.step
                )
                extra_metrics = {
                    "loss_ce": global_avg_ce_loss,
                    "optimizer/lr": last_lr,
                    "optimizer/grad_norm": grad_norm.item(),
                    "optimizer/skipped_step": train_state.skipped_step,
                }
                if dft_enabled:
                    extra_metrics["loss_ce_base"] = global_avg_ce_unweighted_loss
                    extra_metrics["dft_avg_target_prob"] = (
                        global_avg_target_prob
                        if isinstance(global_avg_target_prob, float)
                        else global_avg_target_prob.item()
                    )
                # Log future predictor lr if available (to reflect lr_scale / warmup overrides).
                if future_predictor is not None and fp_idx is not None and hasattr(optimizers, "optimizers"):
                    try:
                        fp_optim = optimizers.optimizers[fp_idx]
                        fp_lr = fp_optim.param_groups[0]["lr"]
                        # Reflect warmup scaling if active.
                        if fp_idx is not None and fp_warmup_steps > 0:
                            fp_lr = fp_optim.param_groups[0]["fp_base_lr"] * fp_warmup_factor
                        extra_metrics["optimizer/lr_future_pred"] = fp_lr
                    except Exception:
                        pass

                if job_config.future_encoder.enable:
                    extra_metrics["aux_loss"] = global_avg_aux_loss
                    # extra_metrics["mi_lower_bound"] = global_avg_mi_lb
                if job_config.action_layer.enable:
                    extra_metrics["action_loss"] = global_avg_action_loss
                    extra_metrics["action_avg_reward"] = global_avg_action_reward
                    extra_metrics["action_avg_size"] = global_avg_action_size
                    extra_metrics["action_max_reward"] = global_max_action_reward

                metric_logger.log(
                    train_state.step,
                    global_avg_loss=global_avg_loss,
                    global_max_loss=global_max_loss,
                    extra_metrics=extra_metrics,
                )

                logger.info(
                    f"{color.blue}lr: {last_lr:.4e} gnorm: {grad_norm:5.2f} "
                    f"{color.magenta}[{str(train_state.elapsed).split('.')[0]:>8}<{str(eta).split('.')[0]:>8}]{color.reset}"
                )

            checkpoint.save(
                train_state.step, force=(train_state.step == job_config.training.steps)
            )

            # signal the profiler that the next profiling step has started
            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step()

            # reduce timeout after first train step for faster signal
            # (assuming lazy init and compilation are finished)
            if train_state.step == 1:
                dist_utils.set_pg_timeouts(
                    timeout=timedelta(seconds=job_config.comm.train_timeout_seconds),
                    world_mesh=world_mesh,
                )

    if torch.distributed.get_rank() == 0:
        logger.info("Sleeping 2 seconds for other ranks to complete")
        time.sleep(2)

    metric_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    init_logger()
    config = JobConfig()
    config.parse_args()
    main(config)
    torch.distributed.destroy_process_group()

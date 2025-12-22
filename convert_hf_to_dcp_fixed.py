"""
Improved HuggingFace to DCP conversion script with parameter validation.

This script addresses potential parameter loss issues by:
1. Using get_model_state_dict for complete state extraction
2. Validating parameter counts before saving
3. Explicit handling of buffers and persistent states
4. Detailed logging of conversion process
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as DCP
from transformers import AutoModelForCausalLM, AutoConfig
from torch.distributed.checkpoint.state_dict import get_model_state_dict

import fla  # noqa
from torchtitan.tools.logging import init_logger, logger
from flame.models.future_encoder import FutureEncoder
from flame.models.action_layer import ActionLayer
from flame.models.future_predictor import FuturePredictorHead


@torch.inference_mode()
def convert_hf_weights_fixed(
    model: str,
    checkpoint: Path,
    trust_remote_code: bool = False,
    use_get_model_state_dict: bool = True,
    include_future_encoder: bool = False,
    future_k: int = 4,
    future_summary_method: str = "mean",
    include_future_predictor: bool = False,
    future_predictor_head_type: str = "linear",
    future_predictor_dropout: float = 0.1,
    include_action_layer: bool = False,
    action_head_type: str = "mlp",
    action_use_rms_norm: bool = False,
    action_ffn_hidden_size: int | None = None,
):
    """
    Convert HuggingFace model weights to DCP format with validation.

    Args:
        use_get_model_state_dict: If True, use get_model_state_dict (recommended).
                                   If False, use model.state_dict() (may be incomplete).
    """
    logger.info(f"Loading model from {model}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model, trust_remote_code=trust_remote_code
    )

    # Method 1: Use get_model_state_dict (recommended)
    if use_get_model_state_dict:
        logger.info("Using get_model_state_dict for complete state extraction")
        state_dict = get_model_state_dict(hf_model)
    else:
        # Method 2: Use traditional state_dict (may miss some buffers)
        logger.info("Using model.state_dict() - may be incomplete")
        state_dict = hf_model.state_dict()

    # Validation: Count parameters
    logger.info("Validating parameter extraction...")

    # Count parameters in the model
    model_param_count = sum(p.numel() for p in hf_model.parameters())
    model_buffer_count = sum(b.numel() for b in hf_model.buffers())

    # Count parameters in state_dict
    state_dict_param_count = sum(
        v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor)
    )

    logger.info(f"  Model parameters:       {model_param_count:,}")
    logger.info(f"  Model buffers:          {model_buffer_count:,}")
    logger.info(f"  State dict tensors:     {state_dict_param_count:,}")
    logger.info(f"  State dict keys:        {len(state_dict)}")

    # Check for parameter loss
    if model_param_count + model_buffer_count != state_dict_param_count:
        diff = abs((model_param_count + model_buffer_count) - state_dict_param_count)
        logger.warning(
            f"⚠️  Parameter count mismatch! Difference: {diff:,} "
            f"({diff / (model_param_count + model_buffer_count) * 100:.2f}%)"
        )

        # Detailed analysis
        param_names = set(name for name, _ in hf_model.named_parameters())
        buffer_names = set(name for name, _ in hf_model.named_buffers())
        state_dict_names = set(state_dict.keys())

        missing_params = param_names - state_dict_names
        missing_buffers = buffer_names - state_dict_names
        extra_keys = state_dict_names - (param_names | buffer_names)

        if missing_params:
            logger.warning(f"  Missing parameters: {sorted(missing_params)}")
        if missing_buffers:
            logger.warning(f"  Missing buffers: {sorted(missing_buffers)}")
        if extra_keys:
            logger.info(f"  Extra keys: {sorted(extra_keys)}")

        logger.warning(
            "  This may cause issues during training. "
            "Consider using use_get_model_state_dict=True"
        )
    else:
        logger.info("  ✅ Parameter counts match!")

    # Log some sample parameters
    logger.info("\nSample parameters in state_dict:")
    for i, (key, value) in enumerate(list(state_dict.items())[:5]):
        if isinstance(value, torch.Tensor):
            logger.info(f"  - {key:60s} {str(value.shape):20s} {value.dtype}")
        else:
            logger.info(f"  - {key:60s} {type(value)}")

    # Infer defaults for action layer
    inferred_action_ffn_hidden_size = action_ffn_hidden_size or getattr(
        hf_model.config, "intermediate_size", None
    )
    inferred_action_use_rms_norm = action_use_rms_norm or bool(
        getattr(hf_model.config, "norm_type", "").lower() == "rmsnorm"
        or hasattr(hf_model.config, "rms_norm_eps")
    )

    # Add optional components
    if include_future_encoder:
        logger.info(
            f"Including FutureEncoder (summary_method={future_summary_method}, future_k={future_k})"
        )
        future_encoder = FutureEncoder(
            hidden_size=hf_model.config.hidden_size,
            future_k=future_k,
            summary_method=future_summary_method,
        )
        fe_state = future_encoder.state_dict()
        logger.info(f"  FutureEncoder parameters: {len(fe_state)}")
        state_dict.update(fe_state)

    if include_future_predictor:
        if not include_future_encoder:
            logger.warning(
                "Including FuturePredictor without FutureEncoder; "
                "ensure future summaries are available during training."
            )
        logger.info(
            f"Including FuturePredictor (head_type={future_predictor_head_type}, "
            f"dropout={future_predictor_dropout})"
        )
        future_predictor = FuturePredictorHead(
            hidden_size=hf_model.config.hidden_size,
            head_type=future_predictor_head_type,
            dropout=future_predictor_dropout,
        )
        fp_state = future_predictor.state_dict()
        logger.info(f"  FuturePredictor parameters: {len(fp_state)}")
        state_dict.update(fp_state)

    if include_action_layer:
        if not include_future_encoder:
            logger.warning(
                "Including ActionLayer without FutureEncoder; "
                "ensure you supply future summaries during training."
            )
        logger.info(
            f"Including ActionLayer (head_type={action_head_type}, "
            f"use_rms_norm={inferred_action_use_rms_norm}, "
            f"ffn_hidden_size={inferred_action_ffn_hidden_size})"
        )
        action_layer = ActionLayer(
            hidden_size=hf_model.config.hidden_size,
            head_type=action_head_type,
            use_rms_norm=inferred_action_use_rms_norm,
            ffn_hidden_size=inferred_action_ffn_hidden_size,
        )
        al_state = action_layer.state_dict()
        logger.info(f"  ActionLayer parameters: {len(al_state)}")
        state_dict.update(al_state)

    # Final validation
    total_tensors = sum(
        v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor)
    )
    logger.info(f"\nFinal state_dict statistics:")
    logger.info(f"  Total keys: {len(state_dict)}")
    logger.info(f"  Total tensor elements: {total_tensors:,}")

    # Save to DCP
    logger.info(f"\nWriting to DCP at '{checkpoint}'")
    checkpoint.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(checkpoint, thread_count=8)

    # Note: Saving with {"model": state_dict} wrapper for DCP compatibility
    # This nested format is required for DCP.load() to work correctly in single-process mode
    DCP.save({"model": state_dict}, storage_writer=storage_writer)

    logger.info("✅ Conversion complete!")
    logger.info(f"   Checkpoint saved to: {checkpoint}")
    logger.info(f"   Use diagnose_checkpoint.py to verify the conversion")


@torch.inference_mode()
def convert_hf_weights_original(
    model: str,
    checkpoint: Path,
    trust_remote_code: bool = False,
    include_future_encoder: bool = False,
    future_k: int = 4,
    future_summary_method: str = "mean",
    include_future_predictor: bool = False,
    future_predictor_head_type: str = "linear",
    future_predictor_dropout: float = 0.1,
    include_action_layer: bool = False,
    action_head_type: str = "mlp",
    action_use_rms_norm: bool = False,
    action_ffn_hidden_size: int | None = None,
):
    """
    Original conversion function (for comparison/fallback).
    """
    logger.info(f"Using ORIGINAL conversion method")
    logger.info(f"Loading model from {model}")
    model = AutoModelForCausalLM.from_pretrained(
        model, trust_remote_code=trust_remote_code
    )
    state_dict = model.state_dict()
    inferred_action_ffn_hidden_size = action_ffn_hidden_size or getattr(
        model.config, "intermediate_size", None
    )
    inferred_action_use_rms_norm = action_use_rms_norm or bool(
        getattr(model.config, "norm_type", "").lower() == "rmsnorm"
        or hasattr(model.config, "rms_norm_eps")
    )

    if include_future_encoder:
        logger.info(
            "Including FutureEncoder parameters (summary_method=%s, future_k=%s)",
            future_summary_method,
            future_k,
        )
        future_encoder = FutureEncoder(
            hidden_size=model.config.hidden_size,
            future_k=future_k,
            summary_method=future_summary_method,
        )
        state_dict.update(future_encoder.state_dict())
        logger.info("FutureEncoder parameters added to checkpoint.")

    if include_future_predictor:
        if not include_future_encoder:
            logger.warning(
                "Including FuturePredictor without FutureEncoder; ensure future summaries are available during training."
            )
        logger.info(
            "Including FuturePredictor parameters (head_type=%s, dropout=%s)",
            future_predictor_head_type,
            future_predictor_dropout,
        )
        future_predictor = FuturePredictorHead(
            hidden_size=model.config.hidden_size,
            head_type=future_predictor_head_type,
            dropout=future_predictor_dropout,
        )
        state_dict.update(future_predictor.state_dict())
        logger.info("FuturePredictor parameters added to checkpoint.")

    if include_action_layer:
        if not include_future_encoder:
            logger.warning(
                "Including ActionLayer without FutureEncoder; ensure you supply future summaries during training."
            )
        logger.info(
            "Including ActionLayer parameters (head_type=%s, use_rms_norm=%s, ffn_hidden_size=%s)",
            action_head_type,
            inferred_action_use_rms_norm,
            inferred_action_ffn_hidden_size,
        )
        action_layer = ActionLayer(
            hidden_size=model.config.hidden_size,
            head_type=action_head_type,
            use_rms_norm=inferred_action_use_rms_norm,
            ffn_hidden_size=inferred_action_ffn_hidden_size,
        )
        state_dict.update(action_layer.state_dict())
        logger.info("ActionLayer parameters added to checkpoint.")

    logger.info(f"Writing to DCP at '{checkpoint}'")
    checkpoint.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(checkpoint, thread_count=8)
    DCP.save({"model": state_dict}, storage_writer=storage_writer)


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser(
        description="Convert huggingface-style model weights to DCP format (FIXED version)."
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code when loading the model.",
    )
    parser.add_argument(
        "--use_original",
        action="store_true",
        help="Use original conversion method (may have parameter loss issues).",
    )
    parser.add_argument(
        "--use_get_model_state_dict",
        action="store_true",
        default=True,
        help="Use get_model_state_dict for complete state extraction (recommended).",
    )
    parser.add_argument(
        "--include_future_encoder",
        action="store_true",
        help="If set, initialize and save a FutureEncoder state_dict alongside the base model.",
    )
    parser.add_argument(
        "--future_k",
        type=int,
        default=4,
        help="future_k argument for the FutureEncoder when --include_future_encoder is enabled.",
    )
    parser.add_argument(
        "--future_summary_method",
        type=str,
        default="mean",
        choices=["mean", "max", "attention"],
        help="FutureEncoder summary_method when --include_future_encoder is enabled.",
    )
    parser.add_argument(
        "--include_future_predictor",
        action="store_true",
        help="If set, initialize and save a FuturePredictorHead state_dict alongside the base model.",
    )
    parser.add_argument(
        "--future_predictor_head_type",
        type=str,
        default="linear",
        choices=["linear", "mlp", "gated"],
        help="Head type for FuturePredictor when --include_future_predictor is enabled.",
    )
    parser.add_argument(
        "--future_predictor_dropout",
        type=float,
        default=0.1,
        help="Dropout rate for FuturePredictor when --include_future_predictor is enabled.",
    )
    parser.add_argument(
        "--include_action_layer",
        action="store_true",
        help="If set, initialize and save an ActionLayer state_dict alongside the base model.",
    )
    parser.add_argument(
        "--action_head_type",
        type=str,
        default="mlp",
        choices=["mlp", "tower"],
        help="ActionLayer head type.",
    )
    parser.add_argument(
        "--action_use_rms_norm",
        action="store_true",
        help="Use RMSNorm inside ActionLayer heads.",
    )
    parser.add_argument(
        "--action_ffn_hidden_size",
        type=int,
        default=None,
        help="Hidden size for ActionLayer FFN (optional).",
    )
    args = parser.parse_args()

    if args.use_original:
        logger.warning("⚠️  Using ORIGINAL conversion method - may have parameter loss issues")
        convert_hf_weights_original(
            args.model,
            args.checkpoint,
            trust_remote_code=args.trust_remote_code,
            include_future_encoder=args.include_future_encoder,
            future_k=args.future_k,
            future_summary_method=args.future_summary_method,
            include_future_predictor=args.include_future_predictor,
            future_predictor_head_type=args.future_predictor_head_type,
            future_predictor_dropout=args.future_predictor_dropout,
            include_action_layer=args.include_action_layer,
            action_head_type=args.action_head_type,
            action_use_rms_norm=args.action_use_rms_norm,
            action_ffn_hidden_size=args.action_ffn_hidden_size,
        )
    else:
        logger.info("✅ Using FIXED conversion method with validation")
        convert_hf_weights_fixed(
            args.model,
            args.checkpoint,
            trust_remote_code=args.trust_remote_code,
            use_get_model_state_dict=args.use_get_model_state_dict,
            include_future_encoder=args.include_future_encoder,
            future_k=args.future_k,
            future_summary_method=args.future_summary_method,
            include_future_predictor=args.include_future_predictor,
            future_predictor_head_type=args.future_predictor_head_type,
            future_predictor_dropout=args.future_predictor_dropout,
            include_action_layer=args.include_action_layer,
            action_head_type=args.action_head_type,
            action_use_rms_norm=args.action_use_rms_norm,
            action_ffn_hidden_size=args.action_ffn_hidden_size,
        )

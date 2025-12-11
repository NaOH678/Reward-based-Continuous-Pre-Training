from __future__ import annotations

# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as DCP
from transformers import AutoModelForCausalLM

import fla  # noqa
from torchtitan.tools.logging import init_logger, logger
from flame.models.future_encoder import FutureEncoder
from flame.models.action_layer import ActionLayer
from flame.models.future_predictor import FuturePredictorHead


@torch.inference_mode()
def convert_hf_weights(
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
    logger.info(f"Loading model from {model}")
    model = AutoModelForCausalLM.from_pretrained(
        model, trust_remote_code=trust_remote_code
    )
    state_dict = model.state_dict()
    # Infer defaults to mirror training when flags are not provided explicitly.
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
    # Save a flat state_dict so it matches TorchTitan CheckpointManager expectations.
    DCP.save(state_dict, storage_writer=storage_writer)


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser(description="Convert huggingface-style model weights to DCP format.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trust_remote_code", action="store_true", help="Whether to trust remote code when loading the model.")
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

    convert_hf_weights(
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

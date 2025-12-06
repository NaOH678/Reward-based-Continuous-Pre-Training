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


@torch.inference_mode()
def convert_hf_weights(
    model: str,
    checkpoint: Path,
    trust_remote_code: bool = False,
    include_future_encoder: bool = False,
    future_k: int = 4,
    future_summary_method: str = "mean",
):
    logger.info(f"Loading model from {model}")
    model = AutoModelForCausalLM.from_pretrained(
        model, trust_remote_code=trust_remote_code
    )
    state_dict = model.state_dict()

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

    logger.info(f"Writing to DCP at '{checkpoint}'")
    checkpoint.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(checkpoint, thread_count=8)
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
    args = parser.parse_args()

    convert_hf_weights(
        args.model,
        args.checkpoint,
        trust_remote_code=args.trust_remote_code,
        include_future_encoder=args.include_future_encoder,
        future_k=args.future_k,
        future_summary_method=args.future_summary_method,
    )

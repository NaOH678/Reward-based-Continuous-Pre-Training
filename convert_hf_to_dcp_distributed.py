"""
Distributed-aware HuggingFace to DCP conversion script.

Key insight: DCP requires a distributed context (torch.distributed) even for
single-process saves. This script uses torchrun to ensure proper DCP operation.

Usage:
    torchrun --nproc_per_node=1 convert_hf_to_dcp_distributed.py \
        --model ../OLMo-1B \
        --checkpoint ../OLMo-1B/checkpoint_correct \
        --trust_remote_code \
        --include_future_predictor \
        --future_predictor_head_type mlp
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as DCP
from transformers import AutoModelForCausalLM

import fla  # noqa
from torchtitan.tools.logging import init_logger, logger
from flame.models.future_encoder import FutureEncoder
from flame.models.action_layer import ActionLayer
from flame.models.future_predictor import FuturePredictorHead


def init_distributed():
    """Initialize distributed environment for DCP."""
    if "RANK" in os.environ:
        # Running under torchrun
        dist.init_process_group(backend="gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        logger.info(f"Initialized distributed: rank={rank}, world_size={world_size}")
        return rank, world_size
    else:
        # Not running under torchrun
        logger.warning(
            "Not running under torchrun. DCP may not work correctly. "
            "Please use: torchrun --nproc_per_node=1 convert_hf_to_dcp_distributed.py ..."
        )
        return 0, 1


@torch.inference_mode()
def convert_hf_to_dcp_model_only(
    model_path: str,
    checkpoint_dir: Path,
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
    Convert HuggingFace model to DCP format compatible with TorchTitan CheckpointManager.

    This matches the format used when CheckpointManager loads with model_only=True,
    which is a FLAT state_dict (not nested under "model" key).
    """
    rank, world_size = init_distributed()

    if rank == 0:
        logger.info("="*80)
        logger.info("HuggingFace to DCP Conversion (Distributed Mode)")
        logger.info("="*80)
        logger.info(f"Model: {model_path}")
        logger.info(f"Checkpoint: {checkpoint_dir}")
        logger.info(f"Trust remote code: {trust_remote_code}")

    # Load model
    if rank == 0:
        logger.info("\nLoading HuggingFace model...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float32  # Use float32 for checkpoint
    )

    if rank == 0:
        logger.info("✅ Model loaded")

    # Get state dict - use standard state_dict() which matches TorchTitan's approach
    state_dict = model.state_dict()

    if rank == 0:
        logger.info(f"\nBase model state_dict:")
        logger.info(f"  Total keys: {len(state_dict)}")
        total_params = sum(v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor))
        logger.info(f"  Total parameters: {total_params:,}")

        # Show sample keys
        logger.info(f"\n  Sample keys:")
        for i, key in enumerate(list(state_dict.keys())[:5]):
            tensor = state_dict[key]
            if isinstance(tensor, torch.Tensor):
                logger.info(f"    {i+1}. {key:60s} {str(tensor.shape):20s} {tensor.dtype}")

    # Infer defaults for action layer
    inferred_action_ffn_hidden_size = action_ffn_hidden_size or getattr(
        model.config, "intermediate_size", None
    )
    inferred_action_use_rms_norm = action_use_rms_norm or bool(
        getattr(model.config, "norm_type", "").lower() == "rmsnorm"
        or hasattr(model.config, "rms_norm_eps")
    )

    # Add optional components
    if include_future_encoder:
        if rank == 0:
            logger.info(
                f"\n✅ Including FutureEncoder (summary_method={future_summary_method}, future_k={future_k})"
            )
        future_encoder = FutureEncoder(
            hidden_size=model.config.hidden_size,
            future_k=future_k,
            summary_method=future_summary_method,
        )
        fe_state = future_encoder.state_dict()
        state_dict.update(fe_state)
        if rank == 0:
            logger.info(f"   Added {len(fe_state)} FutureEncoder parameters")

    if include_future_predictor:
        if not include_future_encoder and rank == 0:
            logger.warning(
                "⚠️  Including FuturePredictor without FutureEncoder; "
                "ensure future summaries are available during training."
            )
        if rank == 0:
            logger.info(
                f"\n✅ Including FuturePredictor (head_type={future_predictor_head_type}, "
                f"dropout={future_predictor_dropout})"
            )
        future_predictor = FuturePredictorHead(
            hidden_size=model.config.hidden_size,
            head_type=future_predictor_head_type,
            dropout=future_predictor_dropout,
        )
        fp_state = future_predictor.state_dict()
        state_dict.update(fp_state)
        if rank == 0:
            logger.info(f"   Added {len(fp_state)} FuturePredictor parameters")

    if include_action_layer:
        if not include_future_encoder and rank == 0:
            logger.warning(
                "⚠️  Including ActionLayer without FutureEncoder; "
                "ensure you supply future summaries during training."
            )
        if rank == 0:
            logger.info(
                f"\n✅ Including ActionLayer (head_type={action_head_type}, "
                f"use_rms_norm={inferred_action_use_rms_norm}, "
                f"ffn_hidden_size={inferred_action_ffn_hidden_size})"
            )
        action_layer = ActionLayer(
            hidden_size=model.config.hidden_size,
            head_type=action_head_type,
            use_rms_norm=inferred_action_use_rms_norm,
            ffn_hidden_size=inferred_action_ffn_hidden_size,
        )
        al_state = action_layer.state_dict()
        state_dict.update(al_state)
        if rank == 0:
            logger.info(f"   Added {len(al_state)} ActionLayer parameters")

    # Final statistics
    if rank == 0:
        total_keys = len(state_dict)
        total_tensors = sum(
            v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor)
        )
        logger.info(f"\n{'='*80}")
        logger.info(f"Final state_dict statistics:")
        logger.info(f"  Total keys: {total_keys}")
        logger.info(f"  Total tensor elements: {total_tensors:,}")
        logger.info(f"{'='*80}")

    # Save using DCP in distributed mode
    # This creates a checkpoint in the format expected by TorchTitan when model_only=True
    # The checkpoint structure will be: checkpoint_dir/step-0/
    step_dir = checkpoint_dir / "step-0"

    if rank == 0:
        logger.info(f"\nSaving to DCP checkpoint: {step_dir}")
        step_dir.mkdir(parents=True, exist_ok=True)

    # Synchronize before saving
    if dist.is_initialized():
        dist.barrier()

    # Save flat state_dict (matches TorchTitan model_only format)
    # When TorchTitan loads with model_only=True, it does:
    #   sd = self.states[MODEL].state_dict()  # flat dict
    #   dcp.load(sd, checkpoint_id=checkpoint_id)
    # So we save a flat state_dict here
    DCP.save(
        state_dict,
        checkpoint_id=str(step_dir),
    )

    if rank == 0:
        logger.info("✅ DCP save completed")

        # Verify the checkpoint was created
        if (step_dir / ".metadata").exists():
            logger.info("✅ Checkpoint metadata file created")
            distcp_files = list(step_dir.glob("*.distcp"))
            logger.info(f"✅ Created {len(distcp_files)} .distcp shard files")
            total_size = sum(f.stat().st_size for f in distcp_files)
            logger.info(f"   Total checkpoint size: {total_size / (1024**3):.2f} GB")
        else:
            logger.error("❌ Checkpoint metadata file not found!")

        logger.info("\n" + "="*80)
        logger.info("Conversion Complete!")
        logger.info("="*80)
        logger.info(f"\nCheckpoint saved to: {step_dir}")
        logger.info("\nTo use this checkpoint in training, set:")
        logger.info(f"  --checkpoint.initial_load_path {step_dir}")
        logger.info(f"  --checkpoint.load_step 0")
        logger.info("\nOr copy it to your checkpoint folder:")
        logger.info(f"  cp -r {step_dir} /path/to/your/checkpoint/folder/")

    # Cleanup
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    init_logger()

    parser = argparse.ArgumentParser(
        description="Convert HuggingFace model to DCP format (distributed mode)"
    )
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model path")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Output checkpoint directory")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code")
    parser.add_argument("--include_future_encoder", action="store_true", help="Include FutureEncoder")
    parser.add_argument("--future_k", type=int, default=4, help="FutureEncoder future_k")
    parser.add_argument(
        "--future_summary_method",
        type=str,
        default="mean",
        choices=["mean", "max", "attention"],
        help="FutureEncoder summary method"
    )
    parser.add_argument("--include_future_predictor", action="store_true", help="Include FuturePredictor")
    parser.add_argument(
        "--future_predictor_head_type",
        type=str,
        default="linear",
        choices=["linear", "mlp", "gated"],
        help="FuturePredictor head type"
    )
    parser.add_argument(
        "--future_predictor_dropout",
        type=float,
        default=0.1,
        help="FuturePredictor dropout"
    )
    parser.add_argument("--include_action_layer", action="store_true", help="Include ActionLayer")
    parser.add_argument(
        "--action_head_type",
        type=str,
        default="mlp",
        choices=["mlp", "tower"],
        help="ActionLayer head type"
    )
    parser.add_argument("--action_use_rms_norm", action="store_true", help="Use RMSNorm in ActionLayer")
    parser.add_argument("--action_ffn_hidden_size", type=int, default=None, help="ActionLayer FFN hidden size")

    args = parser.parse_args()

    convert_hf_to_dcp_model_only(
        model_path=args.model,
        checkpoint_dir=args.checkpoint,
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

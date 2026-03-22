"""
Quick script to check if model weights are random or pretrained.
Useful to verify if checkpoint was actually loaded during training.
"""

import torch
from transformers import AutoModelForCausalLM
import argparse


def check_weights_statistics(model):
    """Check weight statistics to determine if random or pretrained."""
    print("\n" + "="*80)
    print("WEIGHT STATISTICS")
    print("="*80)

    # Check embedding weights
    embed_weight = model.get_input_embeddings().weight
    print(f"\nEmbedding layer:")
    print(f"  Shape: {embed_weight.shape}")
    print(f"  Mean: {embed_weight.float().mean().item():.6f}")
    print(f"  Std: {embed_weight.float().std().item():.6f}")
    print(f"  Min: {embed_weight.float().min().item():.6f}")
    print(f"  Max: {embed_weight.float().max().item():.6f}")

    # Expected values
    print(f"\n  Expected if RANDOM initialization:")
    print(f"    Mean ≈ 0.000, Std ≈ 0.02, Range ≈ [-0.1, 0.1]")
    print(f"\n  Expected if PRETRAINED:")
    print(f"    Mean ≈ 0.000, Std ≈ 0.05-0.15, Range wider")

    # Check first layer weights
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        first_layer = model.model.layers[0]
        if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'q_proj'):
            q_weight = first_layer.self_attn.q_proj.weight
            print(f"\nFirst layer Q projection:")
            print(f"  Shape: {q_weight.shape}")
            print(f"  Mean: {q_weight.float().mean().item():.6f}")
            print(f"  Std: {q_weight.float().std().item():.6f}")
            print(f"  Min: {q_weight.float().min().item():.6f}")
            print(f"  Max: {q_weight.float().max().item():.6f}")

    # Check LM head
    if hasattr(model, 'lm_head'):
        lm_head_weight = model.lm_head.weight
        print(f"\nLM head:")
        print(f"  Shape: {lm_head_weight.shape}")
        print(f"  Mean: {lm_head_weight.float().mean().item():.6f}")
        print(f"  Std: {lm_head_weight.float().std().item():.6f}")
        print(f"  Min: {lm_head_weight.float().min().item():.6f}")
        print(f"  Max: {lm_head_weight.float().max().item():.6f}")

    # Verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)

    embed_std = embed_weight.float().std().item()
    embed_range = (
        embed_weight.float().max().item() - embed_weight.float().min().item()
    )

    if embed_std < 0.03 and embed_range < 0.3:
        print("\n❌ Weights appear to be RANDOMLY INITIALIZED")
        print("   (Small std and narrow range)")
        print("   → Checkpoint was NOT loaded correctly")
    elif embed_std > 0.04 and embed_range > 0.5:
        print("\n✅ Weights appear to be from PRETRAINED model")
        print("   (Larger std and wider range)")
        print("   → Checkpoint was loaded successfully")
    else:
        print("\n⚠️  Uncertain - weights are in between")
        print("   → May need manual inspection")


def main():
    parser = argparse.ArgumentParser(
        description="Check if model weights are random or pretrained"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model (local or hub)"
    )

    args = parser.parse_args()

    print("="*80)
    print("MODEL WEIGHT CHECKER")
    print("="*80)
    print(f"\nLoading model: {args.model}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    print(f"✅ Model loaded")

    check_weights_statistics(model)


if __name__ == "__main__":
    main()

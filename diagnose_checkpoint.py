"""
Diagnostic script to inspect checkpoint format and identify conversion issues.
"""

import argparse
import os
from pathlib import Path
import torch
import torch.distributed.checkpoint as DCP
from transformers import AutoModelForCausalLM, AutoConfig


def inspect_dcp_checkpoint(checkpoint_path: Path):
    """Inspect a DCP checkpoint to see its structure."""
    print(f"\n{'='*80}")
    print(f"Inspecting DCP checkpoint: {checkpoint_path}")
    print(f"{'='*80}\n")

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint path does not exist: {checkpoint_path}")
        return

    # List files in checkpoint
    print("Files in checkpoint:")
    for item in sorted(checkpoint_path.iterdir()):
        size = item.stat().st_size / (1024**2) if item.is_file() else 0
        print(f"  - {item.name:50s} ({size:>10.2f} MB)")

    # Load checkpoint metadata
    try:
        metadata_path = checkpoint_path / ".metadata"
        if metadata_path.exists():
            import pickle
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            print(f"\n✅ Metadata loaded")
            print(f"  Keys in metadata: {list(metadata.keys()) if isinstance(metadata, dict) else 'Not a dict'}")
        else:
            print(f"\n⚠️  No .metadata file found")
    except Exception as e:
        print(f"\n❌ Error loading metadata: {e}")

    # Try to load the checkpoint
    try:
        print(f"\nLoading checkpoint with DCP...")
        state_dict = {}
        DCP.load(state_dict, checkpoint_id=str(checkpoint_path))

        print(f"✅ Checkpoint loaded successfully")
        print(f"\nTop-level keys in checkpoint:")
        for key in sorted(state_dict.keys())[:20]:  # Show first 20 keys
            tensor = state_dict[key]
            if isinstance(tensor, torch.Tensor):
                print(f"  - {key:60s} {str(tensor.shape):20s} {tensor.dtype}")
            else:
                print(f"  - {key:60s} {type(tensor)}")

        if len(state_dict) > 20:
            print(f"  ... and {len(state_dict) - 20} more keys")

        print(f"\nTotal keys: {len(state_dict)}")

        # Check if it has nested structure
        has_model_key = "model" in state_dict
        print(f"\nHas 'model' key: {has_model_key}")

        if has_model_key:
            model_dict = state_dict["model"]
            print(f"  Type of state_dict['model']: {type(model_dict)}")
            if isinstance(model_dict, dict):
                print(f"  Keys in state_dict['model']: {len(model_dict)}")
                for key in list(model_dict.keys())[:5]:
                    print(f"    - {key}")

        # Try to identify parameter prefixes
        print(f"\nParameter prefixes found:")
        prefixes = set()
        for key in state_dict.keys():
            if isinstance(key, str):
                parts = key.split(".")
                if len(parts) > 1:
                    prefixes.add(parts[0])

        for prefix in sorted(prefixes)[:10]:
            count = sum(1 for k in state_dict.keys() if isinstance(k, str) and k.startswith(prefix + "."))
            print(f"  - {prefix:30s} ({count} parameters)")

        return state_dict

    except Exception as e:
        print(f"\n❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_with_hf_model(checkpoint_path: Path, model_name: str, trust_remote_code: bool = False):
    """Compare checkpoint with HuggingFace model to find missing parameters."""
    print(f"\n{'='*80}")
    print(f"Comparing with HuggingFace model: {model_name}")
    print(f"{'='*80}\n")

    # Load HF model
    print(f"Loading HuggingFace model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float32
        )
        hf_state_dict = model.state_dict()
        print(f"✅ HuggingFace model loaded: {len(hf_state_dict)} parameters")
    except Exception as e:
        print(f"❌ Error loading HuggingFace model: {e}")
        return

    # Load DCP checkpoint
    print(f"\nLoading DCP checkpoint...")
    try:
        dcp_state_dict = {}
        DCP.load(dcp_state_dict, checkpoint_id=str(checkpoint_path))

        # Handle nested structure
        if "model" in dcp_state_dict and isinstance(dcp_state_dict["model"], dict):
            dcp_model_dict = dcp_state_dict["model"]
        else:
            dcp_model_dict = dcp_state_dict

        print(f"✅ DCP checkpoint loaded: {len(dcp_model_dict)} parameters")
    except Exception as e:
        print(f"❌ Error loading DCP checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return

    # Compare
    print(f"\n{'='*80}")
    print(f"Comparison Results")
    print(f"{'='*80}\n")

    hf_keys = set(hf_state_dict.keys())
    dcp_keys = set(dcp_model_dict.keys())

    missing_in_dcp = hf_keys - dcp_keys
    extra_in_dcp = dcp_keys - hf_keys
    common_keys = hf_keys & dcp_keys

    print(f"HuggingFace model parameters: {len(hf_keys)}")
    print(f"DCP checkpoint parameters:    {len(dcp_keys)}")
    print(f"Common parameters:            {len(common_keys)}")
    print(f"Missing in DCP:               {len(missing_in_dcp)}")
    print(f"Extra in DCP:                 {len(extra_in_dcp)}")

    if missing_in_dcp:
        print(f"\n⚠️  Parameters MISSING in DCP checkpoint:")
        for key in sorted(missing_in_dcp)[:20]:
            shape = hf_state_dict[key].shape if isinstance(hf_state_dict[key], torch.Tensor) else "N/A"
            print(f"  - {key:60s} {str(shape)}")
        if len(missing_in_dcp) > 20:
            print(f"  ... and {len(missing_in_dcp) - 20} more missing parameters")

    if extra_in_dcp:
        print(f"\n⚠️  Extra parameters in DCP checkpoint (not in HF model):")
        for key in sorted(extra_in_dcp)[:20]:
            tensor = dcp_model_dict[key]
            shape = tensor.shape if isinstance(tensor, torch.Tensor) else "N/A"
            print(f"  - {key:60s} {str(shape)}")
        if len(extra_in_dcp) > 20:
            print(f"  ... and {len(extra_in_dcp) - 20} more extra parameters")

    # Check shape mismatches
    print(f"\nChecking shape mismatches...")
    shape_mismatches = []
    for key in common_keys:
        hf_tensor = hf_state_dict[key]
        dcp_tensor = dcp_model_dict[key]
        if isinstance(hf_tensor, torch.Tensor) and isinstance(dcp_tensor, torch.Tensor):
            if hf_tensor.shape != dcp_tensor.shape:
                shape_mismatches.append((key, hf_tensor.shape, dcp_tensor.shape))

    if shape_mismatches:
        print(f"\n❌ Shape mismatches found: {len(shape_mismatches)}")
        for key, hf_shape, dcp_shape in shape_mismatches[:10]:
            print(f"  - {key:60s}")
            print(f"      HF:  {hf_shape}")
            print(f"      DCP: {dcp_shape}")
    else:
        print(f"✅ No shape mismatches found")

    # Calculate total parameters
    print(f"\nParameter counts:")
    hf_total = sum(p.numel() for p in hf_state_dict.values() if isinstance(p, torch.Tensor))
    dcp_total = sum(p.numel() for p in dcp_model_dict.values() if isinstance(p, torch.Tensor))
    print(f"  HuggingFace: {hf_total:,} parameters")
    print(f"  DCP:         {dcp_total:,} parameters")
    print(f"  Difference:  {abs(hf_total - dcp_total):,} parameters ({abs(hf_total - dcp_total) / hf_total * 100:.2f}%)")

    if abs(hf_total - dcp_total) > 0:
        print(f"\n❌ CRITICAL: Parameter count mismatch! This will cause loss issues.")
    else:
        print(f"\n✅ Parameter counts match")


def main():
    parser = argparse.ArgumentParser(description="Diagnose DCP checkpoint conversion issues")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to DCP checkpoint")
    parser.add_argument("--model", type=str, help="HuggingFace model name for comparison")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code")

    args = parser.parse_args()

    # Inspect checkpoint
    state_dict = inspect_dcp_checkpoint(args.checkpoint)

    # Compare with HF model if specified
    if args.model and state_dict is not None:
        compare_with_hf_model(args.checkpoint, args.model, args.trust_remote_code)

    print(f"\n{'='*80}")
    print("Diagnosis complete")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

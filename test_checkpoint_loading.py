"""
Comprehensive checkpoint loading test to diagnose why parameters appear empty.
Tests multiple loading methods to match training behavior.
"""

import argparse
from pathlib import Path
import torch
import torch.distributed.checkpoint as DCP
from transformers import AutoModelForCausalLM, AutoConfig
import pickle


def test_method1_direct_load(checkpoint_path: Path):
    """Method 1: Direct DCP.load (what diagnose_checkpoint.py does)"""
    print("\n" + "="*80)
    print("METHOD 1: Direct DCP.load (diagnose_checkpoint.py style)")
    print("="*80)

    state_dict = {}
    try:
        DCP.load(state_dict, checkpoint_id=str(checkpoint_path))
        print(f"✅ DCP.load succeeded")
        print(f"   Keys loaded: {len(state_dict)}")

        if len(state_dict) > 0:
            print(f"   Sample keys (first 5):")
            for i, key in enumerate(list(state_dict.keys())[:5]):
                print(f"     - {key}")

            total_params = sum(
                v.numel() for v in state_dict.values()
                if isinstance(v, torch.Tensor)
            )
            print(f"   Total parameters: {total_params:,}")
        else:
            print(f"   ⚠️  State dict is EMPTY!")

        return state_dict

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {}


def test_method2_with_model_template(checkpoint_path: Path, model_path: str):
    """Method 2: Load into model state_dict template"""
    print("\n" + "="*80)
    print("METHOD 2: Load into model state_dict template")
    print("="*80)

    try:
        # Create model
        print("Loading model config...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"✅ Config loaded")

        print("Creating model from config...")
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
        print(f"✅ Model created (meta device)")

        # Get model state dict as template
        model_state = model.state_dict()
        print(f"   Model state_dict keys: {len(model_state)}")

        # Try to load checkpoint into it
        print("Loading checkpoint into model state_dict...")
        DCP.load(model_state, checkpoint_id=str(checkpoint_path))

        # Count non-empty parameters
        loaded_params = sum(
            1 for v in model_state.values()
            if isinstance(v, torch.Tensor) and v.numel() > 0
        )
        print(f"✅ DCP.load completed")
        print(f"   Parameters with data: {loaded_params} / {len(model_state)}")

        # Try to load into model
        print("Loading state_dict into model...")
        missing, unexpected = model.load_state_dict(model_state, strict=False)

        print(f"\n   Load results:")
        print(f"     Missing keys: {len(missing)}")
        print(f"     Unexpected keys: {len(unexpected)}")

        if missing:
            print(f"\n   Sample missing keys (first 10):")
            for key in list(missing)[:10]:
                print(f"     - {key}")

        if unexpected:
            print(f"\n   Sample unexpected keys (first 10):")
            for key in list(unexpected)[:10]:
                print(f"     - {key}")

        return model_state, missing, unexpected

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {}, [], []


def test_method3_check_distcp_files(checkpoint_path: Path):
    """Method 3: Directly inspect .distcp files"""
    print("\n" + "="*80)
    print("METHOD 3: Direct inspection of .distcp files")
    print("="*80)

    distcp_files = sorted(checkpoint_path.glob("*.distcp"))
    print(f"Found {len(distcp_files)} .distcp files")

    if not distcp_files:
        print("❌ No .distcp files found!")
        return

    # Try to read the first file
    first_file = distcp_files[0]
    print(f"\nInspecting: {first_file.name}")
    print(f"  File size: {first_file.stat().st_size / (1024**2):.2f} MB")

    try:
        # Try different unpickling methods
        print("\n  Attempting to unpickle...")
        with open(first_file, 'rb') as f:
            data = pickle.load(f)

        print(f"  ✅ Unpickled successfully")
        print(f"  Data type: {type(data)}")

        if isinstance(data, dict):
            print(f"  Keys in file: {len(data)}")
            if len(data) > 0:
                print(f"  Sample keys (first 5):")
                for key in list(data.keys())[:5]:
                    value = data[key]
                    if isinstance(value, torch.Tensor):
                        print(f"    - {key}: {value.shape} {value.dtype}")
                    else:
                        print(f"    - {key}: {type(value)}")
        elif isinstance(data, list):
            print(f"  List length: {len(data)}")
            if len(data) > 0:
                print(f"  First element type: {type(data[0])}")

    except Exception as e:
        print(f"  ❌ Error reading file: {e}")
        print(f"  File might be in a special DCP format")


def test_method4_check_metadata(checkpoint_path: Path):
    """Method 4: Inspect .metadata file"""
    print("\n" + "="*80)
    print("METHOD 4: Inspect .metadata file")
    print("="*80)

    metadata_file = checkpoint_path / ".metadata"

    if not metadata_file.exists():
        print("❌ No .metadata file found!")
        return

    print(f"Metadata file size: {metadata_file.stat().st_size / 1024:.2f} KB")

    try:
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)

        print(f"✅ Metadata loaded")
        print(f"Type: {type(metadata)}")

        if hasattr(metadata, '__dict__'):
            print(f"\nMetadata attributes:")
            for key, value in metadata.__dict__.items():
                if isinstance(value, (str, int, float, bool)):
                    print(f"  {key}: {value}")
                elif isinstance(value, (list, tuple)):
                    print(f"  {key}: {type(value)} of length {len(value)}")
                elif isinstance(value, dict):
                    print(f"  {key}: dict with {len(value)} keys")
                else:
                    print(f"  {key}: {type(value)}")

        # Check for storage metadata
        if hasattr(metadata, 'storage_data'):
            print(f"\nStorage data found:")
            storage = metadata.storage_data
            print(f"  Type: {type(storage)}")
            if isinstance(storage, dict):
                print(f"  Keys: {list(storage.keys())[:10]}")

        # Check for state dict metadata
        if hasattr(metadata, 'state_dict_metadata'):
            print(f"\nState dict metadata found:")
            sd_meta = metadata.state_dict_metadata
            print(f"  Type: {type(sd_meta)}")
            if isinstance(sd_meta, dict):
                print(f"  Keys count: {len(sd_meta)}")
                print(f"  Sample keys: {list(sd_meta.keys())[:5]}")

    except Exception as e:
        print(f"❌ Error reading metadata: {e}")
        import traceback
        traceback.print_exc()


def test_method5_compare_with_pretrained(checkpoint_path: Path, model_path: str):
    """Method 5: Load pretrained model and compare weights"""
    print("\n" + "="*80)
    print("METHOD 5: Compare with pretrained model weights")
    print("="*80)

    try:
        print("Loading pretrained model (this will download/load full weights)...")
        pretrained_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        pretrained_state = pretrained_model.state_dict()

        print(f"✅ Pretrained model loaded")
        print(f"   Parameters: {len(pretrained_state)}")
        total_pretrained = sum(v.numel() for v in pretrained_state.values())
        print(f"   Total elements: {total_pretrained:,}")

        # Get some weight statistics
        embed_weight = pretrained_model.get_input_embeddings().weight
        print(f"\n   Embedding weight statistics (pretrained):")
        print(f"     Shape: {embed_weight.shape}")
        print(f"     Mean: {embed_weight.float().mean().item():.6f}")
        print(f"     Std: {embed_weight.float().std().item():.6f}")
        print(f"     Min: {embed_weight.float().min().item():.6f}")
        print(f"     Max: {embed_weight.float().max().item():.6f}")

        # Try to load checkpoint into pretrained model
        print(f"\nAttempting to load checkpoint into pretrained model...")
        checkpoint_state = {}
        DCP.load(checkpoint_state, checkpoint_id=str(checkpoint_path))

        if len(checkpoint_state) > 0:
            missing, unexpected = pretrained_model.load_state_dict(
                checkpoint_state, strict=False
            )
            print(f"   Missing: {len(missing)}")
            print(f"   Unexpected: {len(unexpected)}")

            # Check if weights actually changed
            embed_weight_after = pretrained_model.get_input_embeddings().weight
            weight_changed = not torch.allclose(
                embed_weight.float(),
                embed_weight_after.float(),
                rtol=1e-5
            )
            print(f"   Weights changed after load: {weight_changed}")
        else:
            print(f"   ⚠️  Checkpoint state is empty, cannot load")

        return pretrained_state

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {}


def test_method6_storage_reader(checkpoint_path: Path):
    """Method 6: Try using DCP StorageReader directly"""
    print("\n" + "="*80)
    print("METHOD 6: Using DCP StorageReader directly")
    print("="*80)

    try:
        from torch.distributed.checkpoint import FileSystemReader

        storage_reader = FileSystemReader(checkpoint_path)
        print(f"✅ Created FileSystemReader")

        # Try to read metadata
        print("Reading checkpoint metadata...")
        metadata = storage_reader.read_metadata()

        print(f"✅ Metadata read")
        print(f"   Type: {type(metadata)}")

        if hasattr(metadata, 'state_dict_metadata'):
            sd_meta = metadata.state_dict_metadata
            if isinstance(sd_meta, dict):
                print(f"   State dict keys: {len(sd_meta)}")
                print(f"   Sample keys:")
                for key in list(sd_meta.keys())[:5]:
                    print(f"     - {key}")

        # Try to use planner
        from torch.distributed.checkpoint import DefaultLoadPlanner

        state_dict = {}
        planner = DefaultLoadPlanner()

        DCP.load(
            state_dict,
            storage_reader=storage_reader,
            planner=planner
        )

        print(f"\n✅ DCP.load with explicit reader/planner")
        print(f"   Keys loaded: {len(state_dict)}")

        return state_dict

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Test multiple checkpoint loading methods"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to DCP checkpoint directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to HuggingFace model (local or hub)"
    )

    args = parser.parse_args()

    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "CHECKPOINT LOADING TEST SUITE" + " "*29 + "║")
    print("╚" + "="*78 + "╝")

    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Model: {args.model}")

    # List checkpoint files
    print(f"\n{'='*80}")
    print("CHECKPOINT FILES")
    print("="*80)
    if args.checkpoint.exists():
        total_size = 0
        for item in sorted(args.checkpoint.iterdir()):
            size_mb = item.stat().st_size / (1024**2) if item.is_file() else 0
            total_size += size_mb
            print(f"  {item.name:50s} {size_mb:>10.2f} MB")
        print(f"\nTotal size: {total_size:.2f} MB")
    else:
        print(f"❌ Checkpoint directory does not exist!")
        return

    # Run all test methods
    results = {}

    results['method1'] = test_method1_direct_load(args.checkpoint)
    results['method2'] = test_method2_with_model_template(args.checkpoint, args.model)
    test_method3_check_distcp_files(args.checkpoint)
    test_method4_check_metadata(args.checkpoint)
    results['method5'] = test_method5_compare_with_pretrained(args.checkpoint, args.model)
    results['method6'] = test_method6_storage_reader(args.checkpoint)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("\nMethod results:")
    for method, result in results.items():
        if isinstance(result, dict):
            count = len(result)
            print(f"  {method}: {count} keys loaded")
        elif isinstance(result, tuple):
            state, missing, unexpected = result
            print(f"  {method}: {len(state)} keys, {len(missing)} missing, {len(unexpected)} unexpected")

    # Diagnosis
    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)

    method1_empty = len(results.get('method1', {})) == 0

    if method1_empty:
        print("\n❌ ISSUE CONFIRMED: Direct DCP.load returns empty state_dict")
        print("\nPossible causes:")
        print("  1. Checkpoint was saved incorrectly (empty state_dict)")
        print("  2. DCP format version mismatch")
        print("  3. Checkpoint requires special loading procedure")
        print("\n🔧 SOLUTION: Re-convert using convert_hf_to_dcp_fixed.py")
    else:
        print("\n✅ Checkpoint loads successfully")
        print("   The issue might be in how diagnose_checkpoint.py interprets the data")

    print("\n" + "="*80)
    print("Test complete! Check the output above for details.")
    print("="*80)


if __name__ == "__main__":
    main()

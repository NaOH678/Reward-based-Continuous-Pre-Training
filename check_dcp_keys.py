import torch
import torch.distributed as dist
import torch.distributed.checkpoint as DCP
import os
import sys

if "RANK" in os.environ:
    dist.init_process_group(backend="gloo")

checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "../OLMo-1B/checkpoint_nested/step-0"

state_dict = {}
DCP.load(state_dict, checkpoint_id=checkpoint_path)

print(f"Loaded {len(state_dict)} keys from {checkpoint_path}")

# Check for lm_head
has_lm_head = any('lm_head' in k for k in state_dict.keys())
has_embed = any('embed' in k for k in state_dict.keys())

print(f"\nHas lm_head: {has_lm_head}")
print(f"Has embed: {has_embed}")

print("\nAll keys containing 'lm_head':")
for k in state_dict.keys():
    if 'lm_head' in k:
        print(f"  - {k}")

print("\nAll keys containing 'embed':")
for k in state_dict.keys():
    if 'embed' in k:
        print(f"  - {k}")

print("\nFirst 20 keys:")
for k in list(state_dict.keys())[:20]:
    print(f"  - {k}")

if dist.is_initialized():
    dist.destroy_process_group()

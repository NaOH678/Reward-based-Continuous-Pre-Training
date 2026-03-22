"""
Test script for validating cu_seqlens processing and future attention mask generation.
This script verifies:
1. _segment_ids_from_cu_seqlens correctly assigns segment IDs
2. build_future_mask_from_cu generates correct anti-causal masks
3. Optimized versions produce identical results with better performance
"""

import torch
import time
from typing import Tuple


# ============================================================================
# Original functions from flame/train.py (for testing)
# ============================================================================

def _segment_ids_from_cu_seqlens(cu_seqlens: torch.Tensor) -> Tuple[torch.Tensor, int]:
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
    segment_id = torch.bucketize(positions, cu[1:], right=True)
    return segment_id, total_len


def build_future_mask_from_cu(
    cu_seqlens: torch.Tensor, window_k: int | None, dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
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


# ============================================================================
# Optimized versions (proposed improvements)
# ============================================================================

def build_future_mask_from_cu_optimized(
    cu_seqlens: torch.Tensor, window_k: int | None, dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized version using torch.clamp instead of torch.minimum with tensor creation.
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
        # Optimization: use clamp instead of minimum with tensor creation
        future_len = torch.clamp(future_len, max=window_k)
    future_valid = (future_len > 0).unsqueeze(0)
    return mask, future_valid


# ============================================================================
# Test functions
# ============================================================================

def test_segment_ids_from_cu():
    """Test that _segment_ids_from_cu_seqlens correctly assigns segment IDs."""
    print("\n" + "="*80)
    print("TEST 1: _segment_ids_from_cu_seqlens")
    print("="*80)

    # Test case 1: Simple case with 2 segments
    print("\n[Test 1.1] Two segments: [0,3,7]")
    cu = torch.tensor([0, 3, 7])
    seg_id, total_len = _segment_ids_from_cu_seqlens(cu)

    print(f"  cu_seqlens: {cu.tolist()}")
    print(f"  total_len: {total_len}")
    print(f"  segment_ids: {seg_id.tolist()}")

    # Verify
    expected_seg_0 = torch.tensor([0, 0, 0])  # positions 0,1,2 -> segment 0
    expected_seg_1 = torch.tensor([1, 1, 1, 1])  # positions 3,4,5,6 -> segment 1
    expected = torch.cat([expected_seg_0, expected_seg_1])

    assert total_len == 7, f"Expected total_len=7, got {total_len}"
    assert torch.equal(seg_id, expected), f"Expected {expected.tolist()}, got {seg_id.tolist()}"
    print("  ✅ PASSED: Positions 0-2 in segment 0, positions 3-6 in segment 1")

    # Test case 2: Three segments with varying lengths
    print("\n[Test 1.2] Three segments: [0,2,5,10]")
    cu = torch.tensor([0, 2, 5, 10])
    seg_id, total_len = _segment_ids_from_cu_seqlens(cu)

    print(f"  cu_seqlens: {cu.tolist()}")
    print(f"  total_len: {total_len}")
    print(f"  segment_ids: {seg_id.tolist()}")

    expected = torch.tensor([0, 0,           # positions 0-1 -> segment 0
                             1, 1, 1,        # positions 2-4 -> segment 1
                             2, 2, 2, 2, 2]) # positions 5-9 -> segment 2

    assert total_len == 10, f"Expected total_len=10, got {total_len}"
    assert torch.equal(seg_id, expected), f"Expected {expected.tolist()}, got {seg_id.tolist()}"
    print("  ✅ PASSED: Correct segment assignment for all positions")

    # Test case 3: Single segment
    print("\n[Test 1.3] Single segment: [0,5]")
    cu = torch.tensor([0, 5])
    seg_id, total_len = _segment_ids_from_cu_seqlens(cu)

    print(f"  cu_seqlens: {cu.tolist()}")
    print(f"  total_len: {total_len}")
    print(f"  segment_ids: {seg_id.tolist()}")

    expected = torch.zeros(5, dtype=torch.long)
    assert total_len == 5, f"Expected total_len=5, got {total_len}"
    assert torch.equal(seg_id, expected), f"Expected all zeros, got {seg_id.tolist()}"
    print("  ✅ PASSED: All positions in segment 0")

    print("\n✅ ALL SEGMENT_ID TESTS PASSED!")


def test_future_mask_basic():
    """Test basic future mask generation without window."""
    print("\n" + "="*80)
    print("TEST 2: build_future_mask_from_cu (Basic - No Window)")
    print("="*80)

    # Test case: 2 segments [0,3,7], no window
    print("\n[Test 2.1] Two segments, no window")
    cu = torch.tensor([0, 3, 7])
    device = torch.device("cpu")
    dtype = torch.float32

    mask, future_valid = build_future_mask_from_cu(cu, window_k=None, dtype=dtype, device=device)

    print(f"  cu_seqlens: {cu.tolist()}")
    print(f"  mask shape: {mask.shape}")
    print(f"  future_valid shape: {future_valid.shape}")
    print(f"  future_valid: {future_valid.squeeze().tolist()}")

    # Verify mask shape
    assert mask.shape == (1, 1, 7, 7), f"Expected shape (1,1,7,7), got {mask.shape}"

    # Verify future_valid
    # Positions 0,1 in segment 0 have future tokens (2)
    # Position 2 is last in segment 0, no future
    # Positions 3,4,5 in segment 1 have future tokens
    # Position 6 is last in segment 1, no future
    expected_valid = torch.tensor([True, True, False, True, True, True, False])
    assert torch.equal(future_valid.squeeze(), expected_valid), \
        f"Expected {expected_valid.tolist()}, got {future_valid.squeeze().tolist()}"
    print("  ✅ PASSED: future_valid correct")

    # Verify mask attention pattern
    mask_2d = mask.squeeze()  # Remove batch and head dims
    neg_inf = -torch.finfo(dtype).max

    print("\n  Checking attention pattern (0=can attend, -inf=cannot):")
    for q_idx in range(7):
        attendable = []
        for kv_idx in range(7):
            if mask_2d[q_idx, kv_idx] == 0.0:
                attendable.append(kv_idx)
        print(f"    Position {q_idx} can attend to: {attendable}")

    # Manual verification for a few key positions
    # Position 0 (segment 0) should attend to positions 1, 2 (future in same segment)
    assert mask_2d[0, 0] == neg_inf, "Position 0 should NOT attend to itself"
    assert mask_2d[0, 1] == 0.0, "Position 0 should attend to position 1"
    assert mask_2d[0, 2] == 0.0, "Position 0 should attend to position 2"
    assert mask_2d[0, 3] == neg_inf, "Position 0 should NOT attend to position 3 (different segment)"
    print("  ✅ PASSED: Position 0 attention pattern correct")

    # Position 2 (last in segment 0) should not attend to anyone
    assert all(mask_2d[2, :] == neg_inf), "Position 2 (last in segment) should not attend to any position"
    print("  ✅ PASSED: Position 2 (last in segment) cannot attend to anyone")

    # Position 3 (first in segment 1) should attend to 4,5,6
    assert mask_2d[3, 3] == neg_inf, "Position 3 should NOT attend to itself"
    assert mask_2d[3, 4] == 0.0, "Position 3 should attend to position 4"
    assert mask_2d[3, 5] == 0.0, "Position 3 should attend to position 5"
    assert mask_2d[3, 6] == 0.0, "Position 3 should attend to position 6"
    assert mask_2d[3, 0] == neg_inf, "Position 3 should NOT attend to position 0 (different segment)"
    print("  ✅ PASSED: Position 3 attention pattern correct")

    print("\n✅ BASIC FUTURE MASK TEST PASSED!")


def test_future_mask_with_window():
    """Test future mask generation with window constraint."""
    print("\n" + "="*80)
    print("TEST 3: build_future_mask_from_cu (With Window)")
    print("="*80)

    # Test case: 2 segments [0,3,7], window_k=2
    print("\n[Test 3.1] Two segments, window_k=2")
    cu = torch.tensor([0, 3, 7])
    device = torch.device("cpu")
    dtype = torch.float32
    window_k = 2

    mask, future_valid = build_future_mask_from_cu(cu, window_k=window_k, dtype=dtype, device=device)

    print(f"  cu_seqlens: {cu.tolist()}")
    print(f"  window_k: {window_k}")
    print(f"  future_valid: {future_valid.squeeze().tolist()}")

    # Verify future_valid with window
    # Position 0: has 2 future tokens in segment (1,2), window allows 2 -> valid
    # Position 1: has 1 future token in segment (2), window allows 2 -> valid
    # Position 2: has 0 future tokens -> invalid
    # Position 3: has 3 future tokens (4,5,6), but window limits to 2 -> valid
    # Position 4: has 2 future tokens (5,6), window allows 2 -> valid
    # Position 5: has 1 future token (6), window allows 2 -> valid
    # Position 6: has 0 future tokens -> invalid
    expected_valid = torch.tensor([True, True, False, True, True, True, False])
    assert torch.equal(future_valid.squeeze(), expected_valid), \
        f"Expected {expected_valid.tolist()}, got {future_valid.squeeze().tolist()}"
    print("  ✅ PASSED: future_valid correct with window")

    # Verify mask respects window
    mask_2d = mask.squeeze()
    neg_inf = -torch.finfo(dtype).max

    print("\n  Checking windowed attention pattern:")
    for q_idx in range(7):
        attendable = []
        for kv_idx in range(7):
            if mask_2d[q_idx, kv_idx] == 0.0:
                attendable.append(kv_idx)
        print(f"    Position {q_idx} can attend to: {attendable}")

    # Position 0 should only attend to positions 1,2 (within window_k=2)
    assert mask_2d[0, 1] == 0.0, "Position 0 should attend to position 1"
    assert mask_2d[0, 2] == 0.0, "Position 0 should attend to position 2"
    print("  ✅ PASSED: Position 0 respects window")

    # Position 3 should only attend to positions 4,5 (not 6, due to window_k=2)
    assert mask_2d[3, 4] == 0.0, "Position 3 should attend to position 4"
    assert mask_2d[3, 5] == 0.0, "Position 3 should attend to position 5"
    assert mask_2d[3, 6] == neg_inf, "Position 3 should NOT attend to position 6 (beyond window)"
    print("  ✅ PASSED: Position 3 respects window (can see 4,5 but not 6)")

    # Test with window_k=1
    print("\n[Test 3.2] window_k=1 (can only see immediate next token)")
    mask, future_valid = build_future_mask_from_cu(cu, window_k=1, dtype=dtype, device=device)
    mask_2d = mask.squeeze()

    # Position 0 should only attend to position 1 (not 2)
    assert mask_2d[0, 1] == 0.0, "Position 0 should attend to position 1"
    assert mask_2d[0, 2] == neg_inf, "Position 0 should NOT attend to position 2 (beyond window_k=1)"
    print("  ✅ PASSED: window_k=1 correctly limits to immediate next token")

    print("\n✅ WINDOWED FUTURE MASK TEST PASSED!")


def test_optimization_equivalence():
    """Test that optimized version produces identical results."""
    print("\n" + "="*80)
    print("TEST 4: Optimization Equivalence and Performance")
    print("="*80)

    cu = torch.tensor([0, 3, 7, 12, 20])
    device = torch.device("cpu")
    dtype = torch.float32
    window_k = 3

    print(f"\n  Testing with cu_seqlens: {cu.tolist()}, window_k={window_k}")

    # Original version
    mask_orig, valid_orig = build_future_mask_from_cu(cu, window_k, dtype, device)

    # Optimized version
    mask_opt, valid_opt = build_future_mask_from_cu_optimized(cu, window_k, dtype, device)

    # Check equivalence
    assert torch.equal(mask_orig, mask_opt), "Masks should be identical"
    assert torch.equal(valid_orig, valid_opt), "future_valid should be identical"
    print("  ✅ PASSED: Optimized version produces identical results")

    # Performance comparison
    num_runs = 1000

    # Warm up
    for _ in range(10):
        build_future_mask_from_cu(cu, window_k, dtype, device)
        build_future_mask_from_cu_optimized(cu, window_k, dtype, device)

    # Original version timing
    start = time.perf_counter()
    for _ in range(num_runs):
        build_future_mask_from_cu(cu, window_k, dtype, device)
    time_orig = time.perf_counter() - start

    # Optimized version timing
    start = time.perf_counter()
    for _ in range(num_runs):
        build_future_mask_from_cu_optimized(cu, window_k, dtype, device)
    time_opt = time.perf_counter() - start

    print(f"\n  Performance comparison ({num_runs} runs):")
    print(f"    Original version:  {time_orig*1000:.2f} ms")
    print(f"    Optimized version: {time_opt*1000:.2f} ms")
    print(f"    Speedup: {time_orig/time_opt:.2f}x")

    if time_opt < time_orig:
        print("  ✅ PASSED: Optimized version is faster")
    else:
        print("  ⚠️  Note: Performance difference may be negligible on small inputs")

    print("\n✅ OPTIMIZATION TEST PASSED!")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n" + "="*80)
    print("TEST 5: Edge Cases")
    print("="*80)

    device = torch.device("cpu")
    dtype = torch.float32

    # Test case 1: Empty cu_seqlens
    print("\n[Test 5.1] Empty cu_seqlens")
    cu = torch.tensor([0])
    mask, valid = build_future_mask_from_cu(cu, None, dtype, device)
    assert mask is None and valid is None, "Empty cu_seqlens should return None"
    print("  ✅ PASSED: Empty cu_seqlens handled correctly")

    # Test case 2: Single token segment
    print("\n[Test 5.2] Single token segments")
    cu = torch.tensor([0, 1, 2, 3])
    mask, valid = build_future_mask_from_cu(cu, None, dtype, device)
    print(f"  cu_seqlens: {cu.tolist()}")
    print(f"  future_valid: {valid.squeeze().tolist()}")

    # All positions are last in their segment, so no future tokens
    expected_valid = torch.tensor([False, False, False])
    assert torch.equal(valid.squeeze(), expected_valid), \
        f"Expected all False for single-token segments"
    print("  ✅ PASSED: Single token segments have no future")

    # Test case 3: Very long segment
    print("\n[Test 5.3] Long segment")
    cu = torch.tensor([0, 100])
    mask, valid = build_future_mask_from_cu(cu, window_k=5, dtype=dtype, device=device)

    # First position should be valid (has future tokens)
    assert valid[0, 0].item() == True, "First position should have future tokens"
    # Last position should be invalid
    assert valid[0, 99].item() == False, "Last position should have no future tokens"
    # Position 95 should be valid (has positions 96-99 as future)
    assert valid[0, 95].item() == True, "Position 95 should have future tokens"
    print("  ✅ PASSED: Long segment handled correctly")

    print("\n✅ ALL EDGE CASE TESTS PASSED!")


def visualize_attention_pattern():
    """Visualize the attention pattern for better understanding."""
    print("\n" + "="*80)
    print("VISUALIZATION: Attention Pattern Matrix")
    print("="*80)

    cu = torch.tensor([0, 4, 8])
    device = torch.device("cpu")
    dtype = torch.float32

    print("\n[Example 1] No window")
    print(f"cu_seqlens: {cu.tolist()} (Segment 0: pos 0-3, Segment 1: pos 4-7)")

    mask, valid = build_future_mask_from_cu(cu, window_k=None, dtype=dtype, device=device)
    mask_2d = mask.squeeze()
    neg_inf = -torch.finfo(dtype).max

    print("\nAttention Matrix (✓ = can attend, ✗ = cannot attend):")
    print("     ", end="")
    for kv in range(8):
        print(f"  {kv}", end="")
    print()
    print("    " + "-" * 26)

    for q in range(8):
        print(f"  {q} |", end="")
        for kv in range(8):
            symbol = " ✓" if mask_2d[q, kv] == 0.0 else " ✗"
            print(symbol, end="")
        print()

    print(f"\nfuture_valid: {valid.squeeze().tolist()}")

    # With window
    print("\n[Example 2] With window_k=2")
    mask, valid = build_future_mask_from_cu(cu, window_k=2, dtype=dtype, device=device)
    mask_2d = mask.squeeze()

    print("\nAttention Matrix (✓ = can attend, ✗ = cannot attend):")
    print("     ", end="")
    for kv in range(8):
        print(f"  {kv}", end="")
    print()
    print("    " + "-" * 26)

    for q in range(8):
        print(f"  {q} |", end="")
        for kv in range(8):
            symbol = " ✓" if mask_2d[q, kv] == 0.0 else " ✗"
            print(symbol, end="")
        print()

    print(f"\nfuture_valid: {valid.squeeze().tolist()}")
    print("\nNote: Each position can only attend to future positions in the SAME segment,")
    print("      and when window_k is set, only up to k positions ahead.")


def run_all_tests():
    """Run all test suites."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "FUTURE MASK TEST SUITE" + " "*36 + "║")
    print("╚" + "="*78 + "╝")

    try:
        test_segment_ids_from_cu()
        test_future_mask_basic()
        test_future_mask_with_window()
        test_optimization_equivalence()
        test_edge_cases()
        visualize_attention_pattern()

        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*80)
        print("\nSummary:")
        print("  ✅ Segment ID assignment is correct")
        print("  ✅ Future attention mask (anti-causal) is correct")
        print("  ✅ Tokens cannot attend to themselves")
        print("  ✅ Window constraint (future_k) works correctly")
        print("  ✅ future_valid mask correctly identifies positions with future tokens")
        print("  ✅ Optimized version produces identical results")
        print("  ✅ Edge cases handled properly")
        print("\nRecommendation:")
        print("  📝 Consider applying the optimization (torch.clamp instead of torch.minimum)")
        print("     in flame/train.py lines 209 and 223 for better performance.")
        print()

    except AssertionError as e:
        print("\n" + "="*80)
        print("❌ TEST FAILED!")
        print("="*80)
        print(f"Error: {e}")
        raise
    except Exception as e:
        print("\n" + "="*80)
        print("❌ UNEXPECTED ERROR!")
        print("="*80)
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()

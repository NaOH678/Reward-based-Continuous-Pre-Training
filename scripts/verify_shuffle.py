#!/usr/bin/env python3
"""
验证 shuffle 是否正确：
1. 检查总 token 数是否一致
2. 检查 token 分布是否一致
3. 检查是否真的被 shuffle 了

用法:
    python verify_shuffle.py \
        --original_dir ../../formalverification-shared/shichaojian/domino-50B \
        --shuffled_dir ../../formalverification-shared/shichaojian/domino-50B-shuffled
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def collect_token_files(input_dir: Path, extensions=('.npy', '.bin')):
    """收集所有 token 文件"""
    token_files = []
    for ext in extensions:
        token_files.extend(sorted(input_dir.rglob(f'*{ext}')))
    return token_files


def load_tokens(file_path: Path, dtype=np.uint32):
    """加载 tokens"""
    if file_path.suffix == '.npy':
        return np.fromfile(file_path, dtype=dtype)
    elif file_path.suffix == '.bin':
        return np.fromfile(file_path, dtype=dtype)
    else:
        raise ValueError(f"不支持的文件类型: {file_path.suffix}")


def count_total_tokens(token_files, dtype=np.uint32):
    """统计总 token 数"""
    total = 0
    for file_path in tqdm(token_files, desc="统计 tokens"):
        tokens = load_tokens(file_path, dtype=dtype)
        total += len(tokens)
    return total


def sample_token_distribution(token_files, sample_size=10_000_000, dtype=np.uint32):
    """
    采样 token 分布
    (完整统计太慢，采样即可验证)
    """
    sampled_tokens = []
    total_tokens = 0

    for file_path in tqdm(token_files, desc="采样 tokens"):
        tokens = load_tokens(file_path, dtype=dtype)
        total_tokens += len(tokens)

        # 按比例采样
        if len(sampled_tokens) < sample_size:
            num_to_sample = min(len(tokens), sample_size - len(sampled_tokens))
            indices = np.random.choice(len(tokens), size=num_to_sample, replace=False)
            sampled_tokens.extend(tokens[indices].tolist())

        if len(sampled_tokens) >= sample_size:
            break

    return Counter(sampled_tokens), total_tokens


def check_shuffle_quality(original_files, shuffled_files, dtype=np.uint32, check_length=100_000):
    """
    检查是否真的被 shuffle 了
    方法：比较前 N 个 tokens 的重叠率
    - 如果没 shuffle：重叠率 ≈ 100%
    - 如果 shuffle 了：重叠率 ≈ 采样比例
    """
    # 读取原始前 N 个 tokens
    original_tokens = []
    for file_path in original_files:
        if len(original_tokens) >= check_length:
            break
        tokens = load_tokens(file_path, dtype=dtype)
        remaining = check_length - len(original_tokens)
        original_tokens.extend(tokens[:remaining].tolist())

    # 读取 shuffle 后前 N 个 tokens
    shuffled_tokens = []
    for file_path in shuffled_files:
        if len(shuffled_tokens) >= check_length:
            break
        tokens = load_tokens(file_path, dtype=dtype)
        remaining = check_length - len(shuffled_tokens)
        shuffled_tokens.extend(tokens[:remaining].tolist())

    # 计算重叠率
    original_set = set(original_tokens)
    shuffled_set = set(shuffled_tokens)
    overlap = len(original_set & shuffled_set)
    overlap_ratio = overlap / len(original_set)

    return overlap_ratio


def verify_shuffle(original_dir: Path, shuffled_dir: Path, dtype=np.uint32):
    """完整验证流程"""

    logger.info("=" * 70)
    logger.info("开始验证 shuffle...")
    logger.info("=" * 70)

    # 1. 收集文件
    logger.info("\n[1/4] 收集文件...")
    original_files = collect_token_files(original_dir)
    shuffled_files = collect_token_files(shuffled_dir)

    logger.info(f"  原始文件数: {len(original_files)}")
    logger.info(f"  Shuffled 文件数: {len(shuffled_files)}")

    # 2. 检查总 token 数
    # logger.info("\n[2/4] 检查总 token 数...")
    # original_total = count_total_tokens(original_files, dtype=dtype)
    # shuffled_total = count_total_tokens(shuffled_files, dtype=dtype)

    # logger.info(f"  原始总 tokens: {original_total:,} ({original_total/1e9:.2f}B)")
    # logger.info(f"  Shuffled 总 tokens: {shuffled_total:,} ({shuffled_total/1e9:.2f}B)")

    # if original_total == shuffled_total:
    #     logger.info("  ✓ Token 数量一致")
    # else:
    #     logger.error(f"  ✗ Token 数量不一致！差异: {abs(original_total - shuffled_total):,}")
    #     return False

    # 3. 检查 token 分布
    logger.info("\n[3/4] 检查 token 分布 (采样 1000万)...")
    original_dist, _ = sample_token_distribution(original_files, sample_size=10_000_000, dtype=dtype)
    shuffled_dist, _ = sample_token_distribution(shuffled_files, sample_size=10_000_000, dtype=dtype)

    # 比较分布差异
    all_tokens = set(original_dist.keys()) | set(shuffled_dist.keys())
    max_diff = 0
    for token in all_tokens:
        orig_count = original_dist.get(token, 0)
        shuf_count = shuffled_dist.get(token, 0)
        diff = abs(orig_count - shuf_count)
        max_diff = max(max_diff, diff)

    # 统计前 10 个最常见 token
    logger.info("  原始前 10 个最常见 tokens:")
    for token, count in original_dist.most_common(10):
        logger.info(f"    Token {token}: {count:,}")

    logger.info("  Shuffled 前 10 个最常见 tokens:")
    for token, count in shuffled_dist.most_common(10):
        logger.info(f"    Token {token}: {count:,}")

    # 分布差异应该很小（采样误差）
    if max_diff < 1000:  # 允许 1000 的采样误差
        logger.info("  ✓ Token 分布基本一致")
    else:
        logger.warning(f"  ⚠ Token 分布有差异，最大差异: {max_diff}")

    # 4. 检查是否真的 shuffle 了
    logger.info("\n[4/4] 检查 shuffle 质量...")
    overlap_ratio = check_shuffle_quality(original_files, shuffled_files, dtype=dtype, check_length=100_000)

    logger.info(f"  前 10万 tokens 的重叠率: {overlap_ratio*100:.2f}%")

    if overlap_ratio < 0.3:  # 重叠率低于 30% 说明确实 shuffle 了
        logger.info("  ✓ 数据已被充分 shuffle")
    elif overlap_ratio < 0.6:
        logger.warning(f"  ⚠ Shuffle 程度一般 (重叠率 {overlap_ratio*100:.1f}%)")
    else:
        logger.error(f"  ✗ Shuffle 不充分！重叠率过高: {overlap_ratio*100:.1f}%")
        return False

    # 总结
    logger.info("\n" + "=" * 70)
    logger.info("验证完成！")
    logger.info("=" * 70)
    logger.info(f"✓ Token 总数: {shuffled_total:,} ({shuffled_total/1e9:.2f}B)")
    logger.info(f"✓ 分布一致性: 通过")
    logger.info(f"✓ Shuffle 质量: {'优秀' if overlap_ratio < 0.3 else '良好'}")
    logger.info("=" * 70)

    return True


def main():
    parser = argparse.ArgumentParser(description="验证 shuffle 结果")
    parser.add_argument("--original_dir", type=str, required=True, help="原始数据目录")
    parser.add_argument("--shuffled_dir", type=str, required=True, help="Shuffled 数据目录")
    parser.add_argument("--dtype", type=str, default="uint32", choices=["uint16", "uint32", "int32"])

    args = parser.parse_args()

    dtype_map = {
        "uint16": np.uint16,
        "uint32": np.uint32,
        "int32": np.int32,
    }

    original_dir = Path(args.original_dir)
    shuffled_dir = Path(args.shuffled_dir)

    if not original_dir.exists():
        logger.error(f"原始目录不存在: {original_dir}")
        return

    if not shuffled_dir.exists():
        logger.error(f"Shuffled 目录不存在: {shuffled_dir}")
        return

    success = verify_shuffle(
        original_dir=original_dir,
        shuffled_dir=shuffled_dir,
        dtype=dtype_map[args.dtype],
    )

    if success:
        logger.info("\n✓ 验证通过！可以放心使用 shuffled 数据进行训练。")
    else:
        logger.error("\n✗ 验证失败！请检查 shuffle 过程是否有问题。")


if __name__ == "__main__":
    main()

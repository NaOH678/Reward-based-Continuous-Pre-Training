#!/usr/bin/env python3
"""
验证离线 Shuffle 的质量

检查项:
1. Token 总数一致性
2. Token 分布一致性（确保没有修改 token 值）
3. Shuffle 质量（确保数据被充分打乱）
4. 语义完整性（解码样本检查）

用法:
    python scripts/verify_offline_shuffle.py \
        --original_dir /path/to/domino-50B \
        --shuffled_dir /path/to/domino-50B-shuffled \
        --tokenizer_path /path/to/tokenizer \
        --seq_len 4096
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_token_files(input_dir: Path) -> List[Path]:
    """收集所有 token 文件"""
    token_files = []
    for ext in ['.npy', '.bin']:
        token_files.extend(sorted(input_dir.rglob(f'*{ext}')))
    return token_files


def count_tokens(files: List[Path], dtype=np.uint32) -> int:
    """计算总 token 数"""
    total = 0
    for f in tqdm(files, desc="统计 tokens"):
        if f.suffix == '.npy':
            try:
                arr = np.load(f, mmap_mode='r')
                total += arr.shape[0]
            except:
                total += f.stat().st_size // np.dtype(dtype).itemsize
        else:
            total += f.stat().st_size // np.dtype(dtype).itemsize
    return total


def sample_tokens(files: List[Path], num_samples: int = 10_000_000, dtype=np.uint32) -> np.ndarray:
    """采样 tokens 用于分布对比"""
    # 先统计总数
    file_sizes = []
    for f in files:
        if f.suffix == '.npy':
            try:
                arr = np.load(f, mmap_mode='r')
                file_sizes.append((f, arr.shape[0]))
            except:
                file_sizes.append((f, f.stat().st_size // np.dtype(dtype).itemsize))
        else:
            file_sizes.append((f, f.stat().st_size // np.dtype(dtype).itemsize))

    total = sum(s for _, s in file_sizes)

    # 按比例从各文件采样
    samples = []
    for f, size in file_sizes:
        n_samples = int(num_samples * size / total)
        if n_samples == 0:
            continue

        if f.suffix == '.npy':
            try:
                arr = np.fromfile(f, dtype=np.uint32)
            except:
                arr = np.fromfile(f, dtype=dtype)
        else:
            arr = np.fromfile(f, dtype=dtype)

        indices = np.random.choice(len(arr), min(n_samples, len(arr)), replace=False)
        samples.append(arr[indices])

    return np.concatenate(samples)


def check_token_distribution(original_samples: np.ndarray, shuffled_samples: np.ndarray) -> bool:
    """检查 token 分布是否一致"""
    logger.info("检查 token 分布...")

    orig_counter = Counter(original_samples.tolist())
    shuf_counter = Counter(shuffled_samples.tolist())

    # 比较最常见的 tokens
    orig_common = orig_counter.most_common(20)
    shuf_common = shuf_counter.most_common(20)

    logger.info("原始数据 top-10 tokens:")
    for token, count in orig_common[:10]:
        pct = count / len(original_samples) * 100
        logger.info(f"  {token}: {count:,} ({pct:.2f}%)")

    logger.info("Shuffled 数据 top-10 tokens:")
    for token, count in shuf_common[:10]:
        pct = count / len(shuffled_samples) * 100
        logger.info(f"  {token}: {count:,} ({pct:.2f}%)")

    # 检查分布差异
    all_tokens = set(orig_counter.keys()) | set(shuf_counter.keys())
    total_diff = 0
    for token in all_tokens:
        orig_pct = orig_counter.get(token, 0) / len(original_samples)
        shuf_pct = shuf_counter.get(token, 0) / len(shuffled_samples)
        total_diff += abs(orig_pct - shuf_pct)

    logger.info(f"分布差异 (L1): {total_diff:.4f}")

    if total_diff < 0.01:
        logger.info("✓ Token 分布一致")
        return True
    else:
        logger.warning("⚠ Token 分布存在差异")
        return False


def check_shuffle_quality(
    shuffled_dir: Path,
    seq_len: int,
    dtype=np.uint32,
    num_samples: int = 10000
) -> Dict:
    """
    检查 shuffle 质量

    通过分析连续序列来自的域分布来评估
    """
    logger.info("检查 shuffle 质量...")

    # 读取元信息
    metadata_path = shuffled_dir / 'shuffle_metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        logger.info(f"Shuffle 信息:")
        logger.info(f"  类型: {metadata.get('shuffle_type', 'unknown')}")
        logger.info(f"  种子: {metadata.get('seed', 'unknown')}")
        logger.info(f"  总序列: {metadata.get('total_sequences', 'unknown'):,}")

    # 采样连续序列，检查是否存在聚集
    shard_files = sorted(shuffled_dir.glob("shard_*.bin"))

    if not shard_files:
        logger.error("未找到 shard 文件")
        return {}

    # 从第一个 shard 读取连续序列
    first_shard = np.fromfile(shard_files[0], dtype=dtype)
    num_seqs_in_shard = len(first_shard) // seq_len

    logger.info(f"从 {shard_files[0].name} 采样 {min(num_samples, num_seqs_in_shard)} 个连续序列")

    # 计算相邻序列的相似度（简单统计）
    # 如果 shuffle 充分，相邻序列应该来自不同文档
    similarities = []
    for i in range(min(num_samples - 1, num_seqs_in_shard - 1)):
        seq1 = first_shard[i * seq_len : (i + 1) * seq_len]
        seq2 = first_shard[(i + 1) * seq_len : (i + 2) * seq_len]

        # 简单的相似度：首尾 token 匹配率
        # 如果来自同一文档，可能有连续性
        overlap = np.sum(seq1[-100:] == seq2[:100])
        similarities.append(overlap)

    avg_similarity = np.mean(similarities)
    logger.info(f"相邻序列平均重叠: {avg_similarity:.2f} tokens (在 100 token 窗口)")

    if avg_similarity < 5:
        logger.info("✓ Shuffle 质量良好（相邻序列几乎无关联）")
    else:
        logger.warning("⚠ 可能存在聚集（相邻序列有关联）")

    return {
        'avg_similarity': avg_similarity,
        'num_samples': min(num_samples, num_seqs_in_shard),
    }


def check_semantic_integrity(
    shuffled_dir: Path,
    tokenizer_path: str,
    seq_len: int,
    dtype=np.uint32,
    num_samples: int = 5
):
    """
    检查语义完整性

    解码随机序列，验证是否是有意义的文本
    """
    logger.info("检查语义完整性...")

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"无法加载 tokenizer: {e}")
        logger.info("跳过语义完整性检查")
        return

    shard_files = sorted(shuffled_dir.glob("shard_*.bin"))
    if not shard_files:
        logger.error("未找到 shard 文件")
        return

    logger.info(f"随机采样 {num_samples} 个序列并解码:")

    for i in range(num_samples):
        # 随机选择 shard 和位置
        shard = np.random.choice(shard_files)
        tokens = np.fromfile(shard, dtype=dtype)
        num_seqs = len(tokens) // seq_len

        seq_idx = np.random.randint(0, num_seqs)
        sequence = tokens[seq_idx * seq_len : (seq_idx + 1) * seq_len]

        # 解码前 300 个 tokens
        try:
            text = tokenizer.decode(sequence[:300].tolist(), skip_special_tokens=False)
            # 清理显示
            text = text.replace('\n', '\\n')[:200]
        except Exception as e:
            text = f"[解码失败: {e}]"

        logger.info(f"\n=== 样本 {i + 1} (shard={shard.name}, seq={seq_idx}) ===")
        logger.info(text)

    logger.info("")
    logger.info("请检查上述文本是否是有意义的内容（而非乱码）")


def main():
    parser = argparse.ArgumentParser(
        description="验证离线 Shuffle 质量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--original_dir", type=str,
                        help="原始数据目录")
    parser.add_argument("--shuffled_dir", type=str, required=True,
                        help="Shuffled 数据目录")
    parser.add_argument("--tokenizer_path", type=str,
                        help="Tokenizer 路径（用于语义验证）")
    parser.add_argument("--seq_len", type=int, default=4096,
                        help="序列长度 (default: 4096)")
    parser.add_argument("--dtype", type=str, default="uint32",
                        choices=["uint16", "uint32", "int32"])
    parser.add_argument("--skip-distribution", action="store_true",
                        help="跳过分布检查（需要原始数据）")
    parser.add_argument("--skip-semantic", action="store_true",
                        help="跳过语义检查（需要 tokenizer）")

    args = parser.parse_args()

    dtype_map = {
        "uint16": np.uint16,
        "uint32": np.uint32,
        "int32": np.int32,
    }
    dtype = dtype_map[args.dtype]

    shuffled_dir = Path(args.shuffled_dir)

    logger.info("=" * 60)
    logger.info("验证离线 Shuffle")
    logger.info("=" * 60)

    # 1. 检查 token 总数
    logger.info("\n[1] 检查 Shuffled 数据...")
    shuffled_files = collect_token_files(shuffled_dir)
    shuffled_tokens = count_tokens(shuffled_files, dtype)
    logger.info(f"Shuffled tokens: {shuffled_tokens:,}")

    if args.original_dir:
        original_dir = Path(args.original_dir)
        original_files = collect_token_files(original_dir)
        original_tokens = count_tokens(original_files, dtype)
        logger.info(f"Original tokens: {original_tokens:,}")

        # 考虑 seq_len 对齐的损失
        expected_tokens = (original_tokens // args.seq_len) * args.seq_len
        logger.info(f"Expected tokens (aligned): {expected_tokens:,}")

        if shuffled_tokens == expected_tokens:
            logger.info("✓ Token 数量一致")
        else:
            diff = abs(shuffled_tokens - expected_tokens)
            logger.warning(f"⚠ Token 数量差异: {diff:,}")

    # 2. 检查分布
    if args.original_dir and not args.skip_distribution:
        logger.info("\n[2] 检查 Token 分布...")
        original_samples = sample_tokens(original_files, 5_000_000, dtype)
        shuffled_samples = sample_tokens(shuffled_files, 5_000_000, dtype)
        check_token_distribution(original_samples, shuffled_samples)
    else:
        logger.info("\n[2] 跳过分布检查")

    # 3. 检查 shuffle 质量
    logger.info("\n[3] 检查 Shuffle 质量...")
    check_shuffle_quality(shuffled_dir, args.seq_len, dtype)

    # 4. 检查语义完整性
    if args.tokenizer_path and not args.skip_semantic:
        logger.info("\n[4] 检查语义完整性...")
        check_semantic_integrity(shuffled_dir, args.tokenizer_path, args.seq_len, dtype)
    else:
        logger.info("\n[4] 跳过语义检查")

    logger.info("\n" + "=" * 60)
    logger.info("验证完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

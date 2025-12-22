#!/usr/bin/env python3
"""
Sequence-level shuffle 预分词数据集（保持语义完整）

关键：shuffle 序列顺序，不 shuffle 序列内部的 token！

两级 shuffle:
1. File-level: 打乱文件读取顺序（混合不同域）
2. Sequence-level: 打乱序列顺序（局部随机化）

用法示例：
    python shuffle_pretokenized_data.py \
        --input_dir ../../formalverification-shared/shichaojian/domino-50B \
        --output_dir ../../formalverification-shared/shichaojian/domino-50B-shuffled \
        --seq_len 4096 \
        --num_shards 64 \
        --seed 42
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def collect_token_files(input_dir: Path, extensions: Tuple[str, ...] = ('.npy', '.bin')) -> List[Path]:
    """递归收集所有 token 文件"""
    token_files = []
    for ext in extensions:
        token_files.extend(sorted(input_dir.rglob(f'*{ext}')))
    return token_files


def load_tokens(file_path: Path, dtype=np.uint32) -> np.ndarray:
    """从 .npy 或 .bin 文件加载 tokens"""
    if file_path.suffix == '.npy':
        try:
            return np.load(file_path)
        except ValueError:
            # 如果是 raw binary 格式的 .npy
            return np.fromfile(file_path, dtype=dtype)
    elif file_path.suffix == '.bin':
        return np.fromfile(file_path, dtype=dtype)
    else:
        raise ValueError(f"不支持的文件类型: {file_path.suffix}")


def sequence_level_shuffle(
    input_dir: Path,
    output_dir: Path,
    seq_len: int = 4096,
    num_shards: int = 64,
    buffer_size: int = 100000,  # 序列数，不是 token 数
    seed: int = 42,
    dtype=np.uint32,
):
    """
    两级 shuffle (保持语义完整):

    Level 1: 文件级 shuffle
        - 打乱文件读取顺序
        - 混合不同域的数据

    Level 2: 序列级 shuffle
        - 在 buffer 内 shuffle 完整序列
        - 每个序列 (seq_len tokens) 保持完整
        - 不打乱序列内部的 token 顺序！

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        seq_len: 每个序列的长度 (tokens)
        num_shards: 输出 shard 数量 (建议 >= num_workers)
        buffer_size: 序列 buffer 大小 (用于 shuffle)
        seed: 随机种子
        dtype: token 数据类型
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有文件
    logger.info(f"从 {input_dir} 收集 token 文件...")
    token_files = collect_token_files(input_dir)
    logger.info(f"找到 {len(token_files)} 个文件")

    if len(token_files) == 0:
        logger.error("未找到任何 token 文件！")
        return

    # Level 1: 打乱文件顺序
    rng = np.random.RandomState(seed)
    file_indices = np.arange(len(token_files))
    rng.shuffle(file_indices)
    logger.info("✓ Level 1: 文件顺序已打乱")

    # 计算总 token 数和序列数
    logger.info("计算数据规模...")
    total_tokens = 0
    for file_path in tqdm(token_files, desc="扫描文件"):
        tokens = load_tokens(file_path, dtype=dtype)
        total_tokens += len(tokens)

    total_sequences = total_tokens // seq_len
    seqs_per_shard = total_sequences // num_shards

    logger.info(f"  总 tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)")
    logger.info(f"  序列长度: {seq_len}")
    logger.info(f"  总序列数: {total_sequences:,}")
    logger.info(f"  输出 shards: {num_shards}")
    logger.info(f"  每个 shard: ~{seqs_per_shard:,} 序列")
    logger.info(f"  Buffer 大小: {buffer_size} 序列 ({buffer_size * seq_len * 4 / 1e6:.1f} MB)")

    # Level 2: 序列级 shuffle + 写入 shards
    logger.info("✓ Level 2: 开始序列级 shuffle...")

    # 打开所有 shard 文件
    shard_files = []
    for i in range(num_shards):
        shard_path = output_dir / f"shard_{i:04d}.bin"
        shard_files.append(open(shard_path, 'wb'))

    # 序列 buffer (存放完整序列，不是单个 token)
    sequence_buffer = []  # List of np.ndarray, each shape (seq_len,)
    remainder = np.array([], dtype=dtype)  # 跨文件的剩余 tokens

    total_seqs_written = 0
    current_shard = 0
    seqs_in_current_shard = 0

    # 按打乱后的顺序读取文件
    for file_idx in tqdm(file_indices, desc="处理文件"):
        file_path = token_files[file_idx]
        tokens = load_tokens(file_path, dtype=dtype)

        # 合并上一个文件的剩余 tokens
        if len(remainder) > 0:
            tokens = np.concatenate([remainder, tokens])

        # 切分成完整序列
        num_complete_seqs = len(tokens) // seq_len

        for i in range(num_complete_seqs):
            seq = tokens[i * seq_len : (i + 1) * seq_len]
            sequence_buffer.append(seq)

            # Buffer 满了，shuffle 并写入
            if len(sequence_buffer) >= buffer_size:
                # Shuffle 序列顺序（不是 token！）
                rng.shuffle(sequence_buffer)

                # 写入 shards (轮流写入)
                for seq in sequence_buffer:
                    shard_files[current_shard].write(seq.astype(dtype).tobytes())
                    seqs_in_current_shard += 1
                    total_seqs_written += 1

                    # 切换到下一个 shard
                    if seqs_in_current_shard >= seqs_per_shard and current_shard < num_shards - 1:
                        current_shard += 1
                        seqs_in_current_shard = 0

                sequence_buffer = []

        # 保留不完整的部分到下一个文件
        remainder = tokens[num_complete_seqs * seq_len:]

    # 处理最后剩余的 buffer
    if sequence_buffer:
        rng.shuffle(sequence_buffer)
        for seq in sequence_buffer:
            shard_files[current_shard].write(seq.astype(dtype).tobytes())
            seqs_in_current_shard += 1
            total_seqs_written += 1
            if seqs_in_current_shard >= seqs_per_shard and current_shard < num_shards - 1:
                current_shard += 1
                seqs_in_current_shard = 0

    # 关闭所有文件
    for f in shard_files:
        f.close()

    # 输出统计
    logger.info("")
    logger.info("=" * 60)
    logger.info("✓ Shuffle 完成！")
    logger.info("=" * 60)
    logger.info(f"  写入序列: {total_seqs_written:,}")
    logger.info(f"  丢弃 tokens: {len(remainder):,} (不足一个序列)")
    logger.info(f"  输出目录: {output_dir}")
    logger.info("")

    # 验证输出
    logger.info("验证输出文件...")
    total_output_tokens = 0
    for i, shard_path in enumerate(sorted(output_dir.glob("shard_*.bin"))):
        size = shard_path.stat().st_size
        num_tokens = size // np.dtype(dtype).itemsize
        num_seqs = num_tokens // seq_len
        total_output_tokens += num_tokens
        logger.info(f"  {shard_path.name}: {num_seqs:,} 序列, {num_tokens:,} tokens")

    logger.info(f"  总输出 tokens: {total_output_tokens:,}")
    logger.info(f"  预期 tokens: {total_seqs_written * seq_len:,}")

    if total_output_tokens == total_seqs_written * seq_len:
        logger.info("  ✓ 验证通过！")
    else:
        logger.warning("  ⚠ Token 数量不匹配，请检查！")


def main():
    parser = argparse.ArgumentParser(
        description="Sequence-level shuffle（保持语义完整）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基本用法
    python shuffle_pretokenized_data.py \\
        --input_dir /path/to/domino-50B \\
        --output_dir /path/to/domino-50B-shuffled

    # 指定参数
    python shuffle_pretokenized_data.py \\
        --input_dir /path/to/domino-50B \\
        --output_dir /path/to/domino-50B-shuffled \\
        --seq_len 4096 \\
        --num_shards 64 \\
        --buffer_size 100000 \\
        --seed 42
        """
    )
    parser.add_argument("--input_dir", type=str, required=True, help="输入目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--seq_len", type=int, default=4096, help="序列长度 (default: 4096)")
    parser.add_argument("--num_shards", type=int, default=64, help="输出 shard 数量 (default: 64)")
    parser.add_argument("--buffer_size", type=int, default=100000, help="序列 buffer 大小 (default: 100000)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (default: 42)")
    parser.add_argument("--dtype", type=str, default="uint32", choices=["uint16", "uint32", "int32"])

    args = parser.parse_args()

    dtype_map = {
        "uint16": np.uint16,
        "uint32": np.uint32,
        "int32": np.int32,
    }

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        return

    logger.info("=" * 60)
    logger.info("Sequence-level Shuffle（保持语义完整）")
    logger.info("=" * 60)
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"序列长度: {args.seq_len}")
    logger.info(f"Shard 数量: {args.num_shards}")
    logger.info(f"Buffer 大小: {args.buffer_size} 序列")
    logger.info(f"随机种子: {args.seed}")
    logger.info("=" * 60)

    sequence_level_shuffle(
        input_dir=input_dir,
        output_dir=output_dir,
        seq_len=args.seq_len,
        num_shards=args.num_shards,
        buffer_size=args.buffer_size,
        seed=args.seed,
        dtype=dtype_map[args.dtype],
    )


if __name__ == "__main__":
    main()

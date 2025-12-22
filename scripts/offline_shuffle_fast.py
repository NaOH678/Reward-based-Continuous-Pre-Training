#!/usr/bin/env python3
"""
高效离线全局 Shuffle - 顺序读取 + 并行写入

优化策略：
1. 顺序读取所有文件（最大化 I/O 吞吐）
2. 在内存中构建全局索引并排序
3. 多线程并行写入到 memory-mapped 文件
4. 自动验证 shuffle 分布均匀性

关键：避免随机读取！

用法：
    python scripts/offline_shuffle_fast.py \
        --input_dir /path/to/domino-50B \
        --output_dir /path/to/domino-50B-shuffled \
        --seq_len 4096 \
        --num_shards 64 \
        --num_workers 8 \
        --seed 42

时间估算：50B tokens 约 10-20 分钟（取决于存储速度和 workers 数量）
内存需求：约 200MB（索引）+ 读取 buffer
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
from tqdm import tqdm

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_files(input_dir: Path, dtype=np.uint32) -> List[Tuple[Path, int, int]]:
    """
    收集所有文件信息
    返回: [(path, num_tokens, num_sequences), ...]
    """
    token_files = []
    for ext in ['.npy', '.bin']:
        token_files.extend(sorted(input_dir.rglob(f'*{ext}')))

    file_infos = []
    item_size = np.dtype(dtype).itemsize

    for f in tqdm(token_files, desc="扫描文件"):
        try:
            if f.suffix == '.npy':
                with f.open('rb') as fp:
                    magic = fp.read(6)
                if magic == b'\x93NUMPY':
                    arr = np.load(f, mmap_mode='r')
                    num_tokens = arr.shape[0]
                else:
                    num_tokens = f.stat().st_size // item_size
            else:
                num_tokens = f.stat().st_size // item_size

            if num_tokens > 0:
                file_infos.append((f, num_tokens))
        except Exception as e:
            logger.warning(f"跳过 {f}: {e}")

    return file_infos


def offline_shuffle_fast(
    input_dir: Path,
    output_dir: Path,
    seq_len: int = 4096,
    num_shards: int = 64,
    seed: int = 42,
    dtype=np.uint32,
    num_workers: int = 4,
):
    """
    高效离线 shuffle

    策略：
    1. 生成全局置换表：random_position -> (file_idx, local_seq_idx)
    2. 顺序读取每个文件，将序列写入对应的 random_position
    3. 使用内存映射文件作为中间存储
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    item_size = np.dtype(dtype).itemsize

    # ========================================
    # Phase 1: 收集文件信息
    # ========================================
    logger.info("=" * 60)
    logger.info("Phase 1: 收集文件信息")
    logger.info("=" * 60)

    file_infos = collect_files(input_dir, dtype)

    if not file_infos:
        logger.error("未找到有效文件！")
        return

    # 计算每个文件的序列数和全局偏移
    file_seq_info = []  # [(path, num_seqs, global_offset), ...]
    total_sequences = 0
    total_tokens = 0

    for path, num_tokens in file_infos:
        num_seqs = num_tokens // seq_len
        if num_seqs > 0:
            file_seq_info.append((path, num_seqs, total_sequences))
            total_sequences += num_seqs
            total_tokens += num_tokens

    logger.info(f"有效文件: {len(file_seq_info)}")
    logger.info(f"总 tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)")
    logger.info(f"总序列数: {total_sequences:,}")

    # 域统计
    domain_stats = {}
    for path, num_seqs, _ in file_seq_info:
        parts = path.relative_to(input_dir).parts
        domain = parts[0] if len(parts) > 1 else "root"
        domain_stats[domain] = domain_stats.get(domain, 0) + num_seqs

    logger.info("各域分布:")
    for domain, count in sorted(domain_stats.items(), key=lambda x: -x[1]):
        pct = count / total_sequences * 100
        logger.info(f"  {domain}: {count:,} ({pct:.1f}%)")

    # ========================================
    # Phase 2: 生成全局置换
    # ========================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 2: 生成全局置换")
    logger.info("=" * 60)

    rng = np.random.default_rng(seed)

    # 生成置换：shuffled_positions[original_idx] = new_position
    # 即：原始第 i 个序列应该写入到 new_position
    logger.info(f"生成 {total_sequences:,} 个随机位置...")
    shuffled_positions = rng.permutation(total_sequences)

    # 验证置换分布 - 检查 shuffle 后各域在输出中的分布是否均匀
    logger.info("验证 shuffle 分布均匀性...")

    # 构建全局序列索引到域的映射
    seq_to_domain = []
    for path, num_seqs, _ in file_seq_info:
        parts = path.relative_to(input_dir).parts
        domain = parts[0] if len(parts) > 1 else "root"
        seq_to_domain.extend([domain] * num_seqs)

    # 检查输出的前 N 个位置的域分布
    check_size = min(50000, total_sequences)
    # shuffled_positions[i] 表示原始第 i 个序列写入到的位置
    # 我们需要反向映射：位置 j 对应的原始序列是哪个
    inverse_perm = np.argsort(shuffled_positions)

    # 检查前 check_size 个输出位置的域分布
    output_domain_counts: Dict[str, int] = {}
    for out_pos in range(check_size):
        orig_idx = inverse_perm[out_pos]
        domain = seq_to_domain[orig_idx]
        output_domain_counts[domain] = output_domain_counts.get(domain, 0) + 1

    # 计算期望分布
    expected_counts = {
        domain: int(count / total_sequences * check_size)
        for domain, count in domain_stats.items()
    }

    logger.info(f"前 {check_size:,} 个输出位置的域分布 vs 期望:")
    max_deviation = 0.0
    for domain in sorted(domain_stats.keys()):
        actual = output_domain_counts.get(domain, 0)
        expected = expected_counts.get(domain, 0)
        if expected > 0:
            deviation = abs(actual - expected) / expected * 100
            max_deviation = max(max_deviation, deviation)
            logger.info(f"  {domain}: 实际={actual:,}, 期望={expected:,}, 偏差={deviation:.1f}%")

    if max_deviation < 20:
        logger.info(f"✓ Shuffle 分布均匀 (最大偏差 {max_deviation:.1f}%)")
    else:
        logger.warning(f"⚠ Shuffle 分布偏差较大 (最大偏差 {max_deviation:.1f}%)")

    # 清理临时变量
    del seq_to_domain, inverse_perm, output_domain_counts

    # ========================================
    # Phase 3: 创建输出文件（预分配空间）
    # ========================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 3: 预分配输出空间")
    logger.info("=" * 60)

    seqs_per_shard = (total_sequences + num_shards - 1) // num_shards
    bytes_per_seq = seq_len * item_size

    # 计算每个 shard 的实际序列数
    shard_sizes = []
    remaining = total_sequences
    for i in range(num_shards):
        size = min(seqs_per_shard, remaining)
        shard_sizes.append(size)
        remaining -= size

    # 创建内存映射输出文件
    shard_mmaps = []
    for i in range(num_shards):
        shard_path = output_dir / f"shard_{i:04d}.bin"
        size_bytes = shard_sizes[i] * bytes_per_seq

        # 创建文件并预分配空间
        with open(shard_path, 'wb') as f:
            f.seek(size_bytes - 1)
            f.write(b'\0')

        # 打开为内存映射
        mmap = np.memmap(shard_path, dtype=dtype, mode='r+',
                         shape=(shard_sizes[i], seq_len))
        shard_mmaps.append(mmap)

        if i < 3 or i >= num_shards - 2:
            logger.info(f"  shard_{i:04d}.bin: {shard_sizes[i]:,} 序列")
        elif i == 3:
            logger.info("  ...")

    # ========================================
    # Phase 4: 顺序读取，并行随机写入
    # ========================================
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Phase 4: 顺序读取 → 并行随机写入 (workers={num_workers})")
    logger.info("=" * 60)

    # 预计算 shard 查找表（用于快速定位 target_pos 所属的 shard）
    shard_cumsum = np.array([0] + list(np.cumsum(shard_sizes)))

    def find_shard_and_local_pos(target_pos: int) -> Tuple[int, int]:
        """快速查找 target_pos 所属的 shard 和 shard 内位置"""
        shard_idx = np.searchsorted(shard_cumsum[1:], target_pos, side='right')
        local_pos = target_pos - shard_cumsum[shard_idx]
        return shard_idx, local_pos

    # 写入锁（每个 shard 一个锁，避免并发写入冲突）
    shard_locks = [threading.Lock() for _ in range(num_shards)]

    def process_file(file_task: Tuple[int, Path, int, int]) -> int:
        """
        处理单个文件：读取并写入到对应的 shuffle 位置
        返回处理的序列数
        """
        file_idx, path, num_seqs, global_offset = file_task

        # 顺序读取整个文件
        if path.suffix == '.npy':
            try:
                with path.open('rb') as fp:
                    magic = fp.read(6)
                if magic == b'\x93NUMPY':
                    tokens = np.load(path)
                else:
                    tokens = np.fromfile(path, dtype=dtype)
            except:
                tokens = np.fromfile(path, dtype=dtype)
        else:
            tokens = np.fromfile(path, dtype=dtype)

        # 处理这个文件的每个序列
        for local_idx in range(num_seqs):
            global_seq_idx = global_offset + local_idx

            # 获取这个序列的目标位置
            target_pos = shuffled_positions[global_seq_idx]

            # 快速查找目标 shard 和 shard 内位置
            shard_idx, local_pos = find_shard_and_local_pos(target_pos)

            # 读取序列
            start = local_idx * seq_len
            seq = tokens[start:start + seq_len]

            # 写入目标位置（使用锁保护）
            with shard_locks[shard_idx]:
                shard_mmaps[shard_idx][local_pos] = seq

        # 释放内存
        del tokens
        return num_seqs

    # 构建任务列表
    file_tasks = [
        (idx, path, num_seqs, offset)
        for idx, (path, num_seqs, offset) in enumerate(file_seq_info)
    ]

    # 使用线程池并行处理文件
    processed_seqs = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_file, task): task for task in file_tasks}

        with tqdm(total=total_sequences, desc="处理序列", unit="seq") as pbar:
            for future in as_completed(futures):
                try:
                    num_processed = future.result()
                    processed_seqs += num_processed
                    pbar.update(num_processed)
                except Exception as e:
                    task = futures[future]
                    logger.error(f"处理文件 {task[1]} 失败: {e}")

    # 刷新所有 mmap
    logger.info("刷新写入...")
    for mmap in shard_mmaps:
        mmap.flush()
        del mmap

    # ========================================
    # Phase 5: 保存元信息
    # ========================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 5: 保存元信息")
    logger.info("=" * 60)

    metadata = {
        'version': 2,
        'shuffle_type': 'global_permutation_fast',
        'input_dir': str(input_dir.absolute()),
        'seq_len': seq_len,
        'dtype': str(dtype),
        'seed': seed,
        'total_tokens': int(total_sequences * seq_len),
        'total_sequences': int(total_sequences),
        'num_shards': num_shards,
        'num_source_files': len(file_seq_info),
        'domain_stats': {k: int(v) for k, v in domain_stats.items()},
        'shard_sizes': [int(s) for s in shard_sizes],
    }

    with open(output_dir / 'shuffle_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # ========================================
    # 完成
    # ========================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("✓ Shuffle 完成！")
    logger.info("=" * 60)
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"总序列: {total_sequences:,}")
    logger.info(f"")
    logger.info("训练时使用:")
    logger.info(f"  --training.dataset {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="高效离线全局 Shuffle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--num_shards", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="并行处理的线程数")
    parser.add_argument("--dtype", type=str, default="uint32",
                        choices=["uint16", "uint32", "int32"])

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
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("高效离线全局 Shuffle")
    logger.info("=" * 60)
    logger.info(f"输入: {input_dir}")
    logger.info(f"输出: {output_dir}")
    logger.info(f"序列长度: {args.seq_len}")
    logger.info(f"Shards: {args.num_shards}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Workers: {args.num_workers}")
    logger.info("=" * 60)

    offline_shuffle_fast(
        input_dir=input_dir,
        output_dir=output_dir,
        seq_len=args.seq_len,
        num_shards=args.num_shards,
        seed=args.seed,
        dtype=dtype_map[args.dtype],
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()

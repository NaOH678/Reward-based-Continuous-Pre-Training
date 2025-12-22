#!/usr/bin/env python3
"""
离线全局 Shuffle - 基于全局索引置换的物理 shuffle

核心思路：
1. 扫描所有文件，构建全局索引 (file_path, local_chunk_idx)
2. 对索引进行全局置换 (不是 buffer shuffle！)
3. 按置换顺序读取序列，顺序写入新文件

这样可以：
- 实现真正的全局随机，不同域的数据完美混合
- 训练时顺序读取，无 I/O 性能损失
- 每个序列保持语义完整（sequence-level shuffle）

用法：
    python scripts/offline_shuffle.py \
        --input_dir /path/to/domino-50B \
        --output_dir /path/to/domino-50B-shuffled \
        --seq_len 4096 \
        --num_shards 64 \
        --seed 42

时间估算：50B tokens 约 30-60 分钟
"""

import argparse
import json
import mmap
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# 配置日志
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """文件信息"""
    path: Path
    num_tokens: int
    num_sequences: int
    is_standard_npy: bool


def is_standard_npy(file_path: Path) -> bool:
    """检查是否是标准 numpy 格式"""
    with file_path.open('rb') as f:
        magic = f.read(6)
    return magic == b'\x93NUMPY'


def get_file_info(file_path: Path, seq_len: int, dtype=np.uint32) -> Optional[FileInfo]:
    """获取文件信息"""
    try:
        if file_path.suffix == '.npy':
            if is_standard_npy(file_path):
                arr = np.load(file_path, mmap_mode='r')
                num_tokens = arr.shape[0]
                is_npy = True
            else:
                num_tokens = file_path.stat().st_size // np.dtype(dtype).itemsize
                is_npy = False
        elif file_path.suffix == '.bin':
            num_tokens = file_path.stat().st_size // np.dtype(dtype).itemsize
            is_npy = False
        else:
            return None

        num_sequences = num_tokens // seq_len
        if num_sequences == 0:
            return None

        return FileInfo(
            path=file_path,
            num_tokens=num_tokens,
            num_sequences=num_sequences,
            is_standard_npy=is_npy,
        )
    except Exception as e:
        logger.warning(f"跳过文件 {file_path}: {e}")
        return None


def collect_files(input_dir: Path, seq_len: int, dtype=np.uint32) -> List[FileInfo]:
    """收集所有有效的 token 文件"""
    logger.info(f"扫描目录: {input_dir}")

    token_files = []
    for ext in ['.npy', '.bin']:
        token_files.extend(sorted(input_dir.rglob(f'*{ext}')))

    logger.info(f"找到 {len(token_files)} 个文件，正在分析...")

    file_infos = []
    for file_path in tqdm(token_files, desc="分析文件"):
        info = get_file_info(file_path, seq_len, dtype)
        if info is not None:
            file_infos.append(info)

    return file_infos


class SequenceReader:
    """高效的序列读取器，支持文件句柄缓存"""

    def __init__(self, file_infos: List[FileInfo], seq_len: int, dtype=np.uint32, cache_size: int = 32):
        self.file_infos = file_infos
        self.seq_len = seq_len
        self.dtype = dtype
        self.item_size = np.dtype(dtype).itemsize
        self.cache_size = cache_size

        # 文件句柄缓存 (LRU)
        self._mmap_cache: Dict[int, np.ndarray] = {}
        self._cache_order: List[int] = []

    def _get_mmap(self, file_idx: int) -> np.ndarray:
        """获取文件的 mmap，带 LRU 缓存"""
        if file_idx in self._mmap_cache:
            # 移到最近使用
            self._cache_order.remove(file_idx)
            self._cache_order.append(file_idx)
            return self._mmap_cache[file_idx]

        # 需要加载新的
        info = self.file_infos[file_idx]

        if info.is_standard_npy:
            arr = np.load(info.path, mmap_mode='r')
        else:
            arr = np.memmap(info.path, dtype=self.dtype, mode='r')

        # 检查缓存大小
        if len(self._mmap_cache) >= self.cache_size:
            # 移除最久未使用的
            old_idx = self._cache_order.pop(0)
            del self._mmap_cache[old_idx]

        self._mmap_cache[file_idx] = arr
        self._cache_order.append(file_idx)
        return arr

    def read_sequence(self, file_idx: int, local_idx: int) -> np.ndarray:
        """读取单个序列"""
        arr = self._get_mmap(file_idx)
        start = local_idx * self.seq_len
        end = start + self.seq_len
        return np.array(arr[start:end], dtype=self.dtype)

    def close(self):
        """关闭所有缓存"""
        self._mmap_cache.clear()
        self._cache_order.clear()


def build_global_index(file_infos: List[FileInfo]) -> np.ndarray:
    """
    构建全局索引

    返回: shape=(total_sequences, 2) 的数组
          每行是 [file_idx, local_idx]
    """
    total_sequences = sum(info.num_sequences for info in file_infos)
    logger.info(f"构建全局索引: {total_sequences:,} 个序列")

    # 使用 uint32，支持最多 4B 个文件和 4B 个序列
    index = np.zeros((total_sequences, 2), dtype=np.uint32)

    current_idx = 0
    for file_idx, info in enumerate(tqdm(file_infos, desc="构建索引")):
        for local_idx in range(info.num_sequences):
            index[current_idx, 0] = file_idx
            index[current_idx, 1] = local_idx
            current_idx += 1

    return index


def global_shuffle_index(index: np.ndarray, seed: int) -> np.ndarray:
    """
    全局置换索引（关键步骤！）

    这不是 buffer shuffle，而是对整个索引数组做全局随机置换
    """
    logger.info(f"全局置换 {len(index):,} 个索引 (seed={seed})")

    rng = np.random.default_rng(seed)
    shuffled_order = rng.permutation(len(index))

    return index[shuffled_order]


def offline_shuffle(
    input_dir: Path,
    output_dir: Path,
    seq_len: int = 4096,
    num_shards: int = 64,
    seed: int = 42,
    dtype=np.uint32,
    verify: bool = True,
):
    """
    执行离线全局 shuffle

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        seq_len: 序列长度
        num_shards: 输出 shard 数量
        seed: 随机种子
        dtype: token 数据类型
        verify: 是否验证输出
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================
    # Phase 1: 收集文件信息
    # ========================================
    logger.info("=" * 60)
    logger.info("Phase 1: 收集文件信息")
    logger.info("=" * 60)

    file_infos = collect_files(input_dir, seq_len, dtype)

    if not file_infos:
        logger.error("未找到有效的 token 文件！")
        return

    total_tokens = sum(info.num_tokens for info in file_infos)
    total_sequences = sum(info.num_sequences for info in file_infos)

    logger.info(f"有效文件: {len(file_infos)}")
    logger.info(f"总 tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)")
    logger.info(f"总序列数: {total_sequences:,}")
    logger.info(f"序列长度: {seq_len}")

    # 统计各域的数据量
    domain_stats = {}
    for info in file_infos:
        # 尝试从路径中提取域名
        parts = info.path.relative_to(input_dir).parts
        domain = parts[0] if len(parts) > 1 else "root"
        if domain not in domain_stats:
            domain_stats[domain] = 0
        domain_stats[domain] += info.num_sequences

    logger.info("各域数据分布:")
    for domain, count in sorted(domain_stats.items(), key=lambda x: -x[1]):
        pct = count / total_sequences * 100
        logger.info(f"  {domain}: {count:,} 序列 ({pct:.1f}%)")

    # ========================================
    # Phase 2: 构建并置换全局索引
    # ========================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 2: 构建并置换全局索引")
    logger.info("=" * 60)

    index = build_global_index(file_infos)
    shuffled_index = global_shuffle_index(index, seed)

    # 验证置换后的分布（采样前 10000 个）
    if len(shuffled_index) > 10000:
        sample = shuffled_index[:10000]
        sample_domains = []
        for file_idx, _ in sample:
            info = file_infos[file_idx]
            parts = info.path.relative_to(input_dir).parts
            domain = parts[0] if len(parts) > 1 else "root"
            sample_domains.append(domain)

        logger.info("置换后前 10000 个序列的域分布:")
        from collections import Counter
        for domain, count in Counter(sample_domains).most_common():
            pct = count / len(sample_domains) * 100
            expected_pct = domain_stats[domain] / total_sequences * 100
            logger.info(f"  {domain}: {pct:.1f}% (期望 {expected_pct:.1f}%)")

    # ========================================
    # Phase 3: 按置换顺序读取并写入
    # ========================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 3: 按置换顺序写入 shards")
    logger.info("=" * 60)

    seqs_per_shard = (total_sequences + num_shards - 1) // num_shards
    logger.info(f"输出 shards: {num_shards}")
    logger.info(f"每个 shard: ~{seqs_per_shard:,} 序列")

    # 创建序列读取器
    reader = SequenceReader(file_infos, seq_len, dtype, cache_size=64)

    # 打开所有输出文件
    shard_files = []
    for i in range(num_shards):
        shard_path = output_dir / f"shard_{i:04d}.bin"
        shard_files.append(open(shard_path, 'wb'))

    # 写入统计
    shard_seq_counts = [0] * num_shards

    try:
        for i, (file_idx, local_idx) in enumerate(tqdm(shuffled_index, desc="写入数据")):
            # 读取序列
            sequence = reader.read_sequence(file_idx, local_idx)

            # 确定目标 shard
            shard_idx = min(i // seqs_per_shard, num_shards - 1)

            # 写入
            shard_files[shard_idx].write(sequence.tobytes())
            shard_seq_counts[shard_idx] += 1

    finally:
        # 关闭所有文件
        for f in shard_files:
            f.close()
        reader.close()

    # ========================================
    # Phase 4: 保存元信息
    # ========================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 4: 保存元信息")
    logger.info("=" * 60)

    metadata = {
        'version': 2,
        'shuffle_type': 'global_index_permutation',
        'input_dir': str(input_dir.absolute()),
        'seq_len': seq_len,
        'dtype': str(dtype),
        'seed': seed,
        'total_tokens': total_tokens,
        'total_sequences': total_sequences,
        'num_shards': num_shards,
        'num_source_files': len(file_infos),
        'domain_stats': domain_stats,
        'shard_sequence_counts': shard_seq_counts,
    }

    metadata_path = output_dir / 'shuffle_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"元信息已保存: {metadata_path}")

    # ========================================
    # Phase 5: 验证输出
    # ========================================
    if verify:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Phase 5: 验证输出")
        logger.info("=" * 60)

        total_output_tokens = 0
        for i in range(num_shards):
            shard_path = output_dir / f"shard_{i:04d}.bin"
            size = shard_path.stat().st_size
            num_tokens = size // np.dtype(dtype).itemsize
            total_output_tokens += num_tokens

            if i < 5 or i >= num_shards - 2:
                logger.info(f"  shard_{i:04d}.bin: {shard_seq_counts[i]:,} 序列, {num_tokens:,} tokens")
            elif i == 5:
                logger.info(f"  ...")

        expected_tokens = total_sequences * seq_len
        logger.info(f"")
        logger.info(f"  输出 tokens: {total_output_tokens:,}")
        logger.info(f"  期望 tokens: {expected_tokens:,}")

        if total_output_tokens == expected_tokens:
            logger.info(f"  ✓ 验证通过！")
        else:
            logger.error(f"  ✗ Token 数量不匹配！")

    # ========================================
    # 完成
    # ========================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("✓ 离线 Shuffle 完成！")
    logger.info("=" * 60)
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"Shard 数量: {num_shards}")
    logger.info(f"总序列: {total_sequences:,}")
    logger.info(f"")
    logger.info("训练时使用:")
    logger.info(f"  --training.dataset {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="离线全局 Shuffle（基于全局索引置换）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基本用法
    python scripts/offline_shuffle.py \\
        --input_dir /path/to/domino-50B \\
        --output_dir /path/to/domino-50B-shuffled

    # 指定参数
    python scripts/offline_shuffle.py \\
        --input_dir /path/to/domino-50B \\
        --output_dir /path/to/domino-50B-shuffled \\
        --seq_len 4096 \\
        --num_shards 64 \\
        --seed 42

核心原理:
    1. 扫描所有文件，构建全局索引 [(file_idx, local_idx), ...]
    2. 对索引进行全局随机置换（不是 buffer shuffle！）
    3. 按置换顺序读取序列，顺序写入新文件

    这样可以实现真正的全局随机，不同域的数据完美混合。
        """
    )

    parser.add_argument("--input_dir", type=str, required=True,
                        help="输入目录（包含 .npy/.bin token 文件）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--seq_len", type=int, default=4096,
                        help="序列长度 (default: 4096)")
    parser.add_argument("--num_shards", type=int, default=64,
                        help="输出 shard 数量 (default: 64)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (default: 42)")
    parser.add_argument("--dtype", type=str, default="uint32",
                        choices=["uint16", "uint32", "int32", "int64"],
                        help="Token 数据类型 (default: uint32)")
    parser.add_argument("--no-verify", action="store_true",
                        help="跳过输出验证")

    args = parser.parse_args()

    dtype_map = {
        "uint16": np.uint16,
        "uint32": np.uint32,
        "int32": np.int32,
        "int64": np.int64,
    }

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        sys.exit(1)

    if output_dir.exists() and any(output_dir.iterdir()):
        logger.warning(f"输出目录非空: {output_dir}")
        response = input("继续将覆盖现有文件，是否继续? [y/N] ")
        if response.lower() != 'y':
            logger.info("已取消")
            sys.exit(0)

    logger.info("=" * 60)
    logger.info("离线全局 Shuffle")
    logger.info("=" * 60)
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"序列长度: {args.seq_len}")
    logger.info(f"Shard 数量: {args.num_shards}")
    logger.info(f"随机种子: {args.seed}")
    logger.info(f"数据类型: {args.dtype}")
    logger.info("=" * 60)

    offline_shuffle(
        input_dir=input_dir,
        output_dir=output_dir,
        seq_len=args.seq_len,
        num_shards=args.num_shards,
        seed=args.seed,
        dtype=dtype_map[args.dtype],
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()

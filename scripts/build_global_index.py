#!/usr/bin/env python3
"""
构建全局索引文件 - OLMo 风格的离线 shuffle 方案

核心思路：
1. 扫描所有 token 文件，建立全局索引映射
2. 索引格式：global_idx -> (file_idx, local_chunk_idx)
3. 训练时按 seed+epoch 生成全局置换，实现真正的全局 shuffle

用法：
    python scripts/build_global_index.py \
        --input_dir /path/to/tokenized_data \
        --output_path /path/to/index.bin \
        --seq_len 4096

优势：
- 不需要复制/移动数据，只生成索引
- 内存占用极低（只存储索引）
- 支持 epoch 级别 reshuffle
- 完全随机的全局 shuffle
"""

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict
import logging
import struct

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def collect_token_files(input_dir: Path, extensions: Tuple[str, ...] = ('.npy', '.bin')) -> List[Path]:
    """递归收集所有 token 文件"""
    token_files = []
    for ext in extensions:
        token_files.extend(sorted(input_dir.rglob(f'*{ext}')))
    return sorted(token_files, key=lambda p: str(p))


def get_file_token_count(file_path: Path, dtype=np.uint32) -> int:
    """获取文件中的 token 数量"""
    if file_path.suffix == '.npy':
        # 检查是否是标准 npy 格式
        with file_path.open('rb') as f:
            magic = f.read(6)
        if magic == b'\x93NUMPY':
            arr = np.load(file_path, mmap_mode='r')
            return arr.shape[0]
        else:
            # raw binary 格式的 .npy
            return file_path.stat().st_size // np.dtype(dtype).itemsize
    elif file_path.suffix == '.bin':
        return file_path.stat().st_size // np.dtype(dtype).itemsize
    else:
        raise ValueError(f"不支持的文件类型: {file_path.suffix}")


def build_global_index(
    input_dir: Path,
    output_path: Path,
    seq_len: int = 4096,
    dtype=np.uint32,
) -> Dict:
    """
    构建全局索引文件

    索引格式 (二进制):
    - Header (JSON): 元信息
    - Index array: uint64 数组，每个元素 = (file_idx << 32) | local_chunk_idx

    Args:
        input_dir: 输入目录
        output_path: 输出索引文件路径
        seq_len: 序列长度
        dtype: token 数据类型

    Returns:
        统计信息字典
    """
    logger.info(f"从 {input_dir} 收集 token 文件...")
    token_files = collect_token_files(input_dir)
    logger.info(f"找到 {len(token_files)} 个文件")

    if len(token_files) == 0:
        logger.error("未找到任何 token 文件！")
        return None

    # 第一遍扫描：计算每个文件的 chunk 数
    logger.info("扫描文件，计算索引...")
    file_infos = []
    total_tokens = 0
    total_chunks = 0

    for file_idx, file_path in enumerate(tqdm(token_files, desc="扫描文件")):
        num_tokens = get_file_token_count(file_path, dtype)
        num_chunks = num_tokens // seq_len  # drop_last=True

        if num_chunks > 0:
            file_infos.append({
                'file_idx': file_idx,
                'path': str(file_path.relative_to(input_dir)),
                'num_tokens': num_tokens,
                'num_chunks': num_chunks,
                'chunk_offset': total_chunks,
            })
            total_tokens += num_tokens
            total_chunks += num_chunks

    logger.info(f"  总 tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)")
    logger.info(f"  序列长度: {seq_len}")
    logger.info(f"  总 chunks: {total_chunks:,}")
    logger.info(f"  有效文件: {len(file_infos)}")

    # 构建全局索引数组
    # 格式: global_idx -> (file_idx << 32) | local_chunk_idx
    logger.info("构建全局索引数组...")
    global_index = np.zeros(total_chunks, dtype=np.uint64)

    for info in tqdm(file_infos, desc="构建索引"):
        file_idx = info['file_idx']
        chunk_offset = info['chunk_offset']
        num_chunks = info['num_chunks']

        for local_idx in range(num_chunks):
            global_idx = chunk_offset + local_idx
            # 编码: high 32 bits = file_idx, low 32 bits = local_chunk_idx
            global_index[global_idx] = (file_idx << 32) | local_idx

    # 保存索引文件
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 元信息
    metadata = {
        'version': 1,
        'input_dir': str(input_dir.absolute()),
        'seq_len': seq_len,
        'dtype': str(dtype),
        'total_tokens': total_tokens,
        'total_chunks': total_chunks,
        'num_files': len(file_infos),
        'files': file_infos,
    }

    # 写入索引文件
    metadata_json = json.dumps(metadata, indent=2).encode('utf-8')

    with output_path.open('wb') as f:
        # Header: 8 bytes length + JSON
        f.write(struct.pack('<Q', len(metadata_json)))
        f.write(metadata_json)
        # Index array
        global_index.tofile(f)

    # 同时保存一个可读的 JSON 元信息文件
    json_path = output_path.with_suffix('.json')
    with json_path.open('w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info("✓ 索引构建完成！")
    logger.info("=" * 60)
    logger.info(f"  索引文件: {output_path}")
    logger.info(f"  元信息文件: {json_path}")
    logger.info(f"  索引大小: {output_path.stat().st_size / 1e6:.1f} MB")
    logger.info(f"  总 chunks: {total_chunks:,}")

    return metadata


def verify_index(index_path: Path) -> bool:
    """验证索引文件"""
    logger.info(f"验证索引文件: {index_path}")

    with index_path.open('rb') as f:
        # 读取元信息
        header_len = struct.unpack('<Q', f.read(8))[0]
        metadata = json.loads(f.read(header_len).decode('utf-8'))

        # 读取索引数组
        global_index = np.fromfile(f, dtype=np.uint64)

    logger.info(f"  版本: {metadata['version']}")
    logger.info(f"  总 chunks: {metadata['total_chunks']}")
    logger.info(f"  索引数组长度: {len(global_index)}")

    if len(global_index) != metadata['total_chunks']:
        logger.error("  ❌ 索引数组长度不匹配！")
        return False

    # 验证编码
    sample_indices = [0, len(global_index) // 2, len(global_index) - 1]
    for idx in sample_indices:
        encoded = global_index[idx]
        file_idx = encoded >> 32
        local_idx = encoded & 0xFFFFFFFF
        logger.info(f"  样本 {idx}: file_idx={file_idx}, local_idx={local_idx}")

    logger.info("  ✓ 索引验证通过！")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="构建全局索引文件（OLMo 风格）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 构建索引
    python scripts/build_global_index.py \\
        --input_dir /path/to/domino-50B \\
        --output_path /path/to/domino-50B.index.bin \\
        --seq_len 4096

    # 验证索引
    python scripts/build_global_index.py \\
        --verify /path/to/domino-50B.index.bin
        """
    )
    parser.add_argument("--input_dir", type=str, help="输入目录")
    parser.add_argument("--output_path", type=str, help="输出索引文件路径")
    parser.add_argument("--seq_len", type=int, default=4096, help="序列长度 (default: 4096)")
    parser.add_argument("--dtype", type=str, default="uint32", choices=["uint16", "uint32", "int32"])
    parser.add_argument("--verify", type=str, help="验证已有索引文件")

    args = parser.parse_args()

    dtype_map = {
        "uint16": np.uint16,
        "uint32": np.uint32,
        "int32": np.int32,
    }

    if args.verify:
        verify_index(Path(args.verify))
        return

    if not args.input_dir or not args.output_path:
        parser.error("需要 --input_dir 和 --output_path 参数")

    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)

    if not input_dir.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        return

    logger.info("=" * 60)
    logger.info("构建全局索引（OLMo 风格）")
    logger.info("=" * 60)
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"输出路径: {output_path}")
    logger.info(f"序列长度: {args.seq_len}")
    logger.info("=" * 60)

    build_global_index(
        input_dir=input_dir,
        output_path=output_path,
        seq_len=args.seq_len,
        dtype=dtype_map[args.dtype],
    )


if __name__ == "__main__":
    main()

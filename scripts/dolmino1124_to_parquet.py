#!/usr/bin/env python
"""
流式混合 allenai/dolmino-mix-1124，按权重累计约 50B token，写本地 Parquet（text, source）。
偏重速度：批量处理，支持长度估算模式。
"""
from __future__ import annotations

import math
import random
from itertools import islice
from pathlib import Path
from typing import Dict, Iterator, List

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

TOTAL_TOKENS = 50_000_000_000  # 目标总 token
TOKENIZER = "allenai/dolma2-tokenizer"
OUT_DIR = Path("data/dolmino1124_mix50b_parquet")
SHARD_TOKENS = 200_000_000  # 每片目标 token（粗计），调大减少写盘
BATCH_SIZE = 8192  # 批量分词/估算的样本数，越大越快但占内存
SHUFFLE_BUFFER = 10_000  # streaming 下的洗牌缓冲区（越大越随机，越占内存）
SHUFFLE_SEED = 1337

# TOKENIZE_MODE:
#   "encode"     精确分词计数（慢但准）
#   "length_est" 长度估算 token 数（快，需设置 avg）
TOKENIZE_MODE = "length_est"
AVG_CHARS_PER_TOKEN = 4.0
AUTO_ESTIMATE_AVG = True
ESTIMATE_SAMPLES = 400
ESTIMATE_SOURCE = "dclm"

# 是否按训练 chunk 取整计耗（True 时消耗=ceil(tok_len/chunk)*chunk）
USE_BLOCK_SIZE = False
BLOCK_SIZE = 4096

# 权重/子源名
WEIGHTS: Dict[str, float] = {
    "dclm": 0.472,
    "flan": 0.166,
    "pes2o": 0.0585,
    "wiki": 0.0711,
    "se": 0.0245,  # stackexchange
    "math": 0.208,
}
NAME_MAP: Dict[str, str] = {
    "dclm": "dclm",
    "flan": "flan",
    "pes2o": "pes2o",
    "wiki": "wiki",
    "se": "stackexchange",
    "math": "math",
}


def stream_sources() -> Dict[str, Iterator]:
    streams = {}
    for k, name in NAME_MAP.items():
        ds = load_dataset("allenai/dolmino-mix-1124", name, split="train", streaming=True)
        ds = ds.shuffle(buffer_size=SHUFFLE_BUFFER, seed=SHUFFLE_SEED)
        streams[k] = iter(ds)
    return streams


def estimate_avg_chars_per_token(tk: AutoTokenizer) -> float:
    """用少量样本粗估 chars/token，比默认常数更准，但不会大幅占用时间。"""
    src_name = NAME_MAP.get(ESTIMATE_SOURCE, ESTIMATE_SOURCE)
    ds = load_dataset("allenai/dolmino-mix-1124", src_name, split="train", streaming=True)
    texts = []
    for ex in islice(ds, ESTIMATE_SAMPLES):
        t = ex.get("text") or ""
        if t:
            texts.append(t)
    if not texts:
        return AVG_CHARS_PER_TOKEN
    ids_batch = tk(
        texts,
        add_special_tokens=False,
        return_attention_mask=False,
        truncation=False,
    )["input_ids"]
    chars = sum(len(t) for t in texts)
    toks = sum(len(ids) for ids in ids_batch) or 1
    return chars / toks


def flush_shard(
    shard_id: int, rows_text: List[str], rows_src: List[str], out_dir: Path
) -> None:
    if not rows_text:
        return
    table = pa.Table.from_arrays(
        [pa.array(rows_text), pa.array(rows_src)],
        names=["text", "source"],
    )
    pq.write_table(table, out_dir / f"part-{shard_id:05d}.parquet", compression="snappy")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tk = AutoTokenizer.from_pretrained(TOKENIZER)
    tk.model_max_length = int(1e9)  # 防止过长样本触发截断警告

    avg_chars_per_token = AVG_CHARS_PER_TOKEN
    if TOKENIZE_MODE == "length_est" and AUTO_ESTIMATE_AVG:
        avg_chars_per_token = estimate_avg_chars_per_token(tk)
        print(f"[length_est] estimated avg_chars_per_token={avg_chars_per_token:.3f} "
              f"from {ESTIMATE_SAMPLES} samples of {ESTIMATE_SOURCE}")

    quota = {k: int(v * TOTAL_TOKENS) for k, v in WEIGHTS.items()}
    used = {k: 0 for k in WEIGHTS}
    total_used = 0
    active = dict(WEIGHTS)
    streams = stream_sources()

    shard_id = 0
    shard_tokens = 0
    rows_text: List[str] = []
    rows_src: List[str] = []
    pbar = tqdm(total=TOTAL_TOKENS, unit="tok")

    while active and total_used < TOTAL_TOKENS:
        # 拉一批原文
        pending_src, pending_text = [], []
        while len(pending_text) < BATCH_SIZE and active and total_used < TOTAL_TOKENS:
            keys, probs = zip(*active.items())
            src = random.choices(keys, probs, k=1)[0]
            try:
                ex = next(streams[src])
            except StopIteration:
                active.pop(src, None)
                continue
            text = ex.get("text") or ""
            if not text:
                continue
            pending_src.append(src)
            pending_text.append(text)

        if not pending_text:
            break

        # 计算 token 长度
        if TOKENIZE_MODE == "encode":
            enc = tk(
                pending_text,
                add_special_tokens=False,
                return_attention_mask=False,
                truncation=False,
            )["input_ids"]
            lens = [len(ids) + 1 for ids in enc]
        elif TOKENIZE_MODE == "length_est":
            lens = [int(len(t) / avg_chars_per_token) + 1 for t in pending_text]
        else:
            raise ValueError(f"Unknown TOKENIZE_MODE: {TOKENIZE_MODE}")

        # 更新配额并写缓存
        for src, tok_len, text in zip(pending_src, lens, pending_text):
            consumed = (
                math.ceil(tok_len / BLOCK_SIZE) * BLOCK_SIZE if USE_BLOCK_SIZE else tok_len
            )

            rows_text.append(text)
            rows_src.append(src)
            shard_tokens += consumed
            used[src] += consumed
            total_used += consumed
            pbar.update(consumed)

            if used[src] >= quota[src]:
                active.pop(src, None)

            if shard_tokens >= SHARD_TOKENS:
                flush_shard(shard_id, rows_text, rows_src, OUT_DIR)
                shard_id += 1
                shard_tokens = 0
                rows_text, rows_src = [], []

    # 写最后一片
    flush_shard(shard_id, rows_text, rows_src, OUT_DIR)
    pbar.close()
    print(f"done, total_used={total_used:,}, shards={shard_id + 1}")


if __name__ == "__main__":
    main()

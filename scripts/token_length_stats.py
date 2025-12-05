#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import itertools
import os
from typing import List, Optional

from transformers import AutoTokenizer

from flame.data import build_dataset
from torchtitan.tools import utils

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _conversation_to_text(sample):
    conversations = sample.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError(f"Expected 'conversations' to be a list, got {type(conversations)}")
    lines = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        value = turn.get("value")
        if value is None:
            continue
        speaker = turn.get("from")
        prefix = f"{speaker}: " if speaker else ""
        lines.append(f"{prefix}{value}")
    metadata = [
        f"{k}={sample[k]}"
        for k in ("difficulty", "source", "domain")
        if k in sample and sample[k] not in (None, "")
    ]
    text_parts = metadata + lines
    return {"text": "\n\n".join(str(p) for p in text_parts if p is not None)}


def _extract_text(sample):
    if sample.get("text") is not None:
        return sample["text"]
    if sample.get("content") is not None:
        return sample["content"]
    if sample.get("conversations") is not None:
        return _conversation_to_text(sample)["text"]
    raise ValueError(f"No 'text', 'content', or 'conversations' field found in sample:\n{sample}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute token length stats for samples.")
    parser.add_argument(
        "--dataset",
        required=False,
        help="Dataset path/name (same as training arg). Required unless --parquet_file is set.",
    )
    parser.add_argument("--tokenizer_path", required=True, help="Tokenizer path/name.")
    parser.add_argument("--dataset_name", default=None, help="Optional dataset config name.")
    parser.add_argument("--dataset_split", default="train", help="Dataset split, default: train.")
    parser.add_argument("--data_dir", default=None, help="Optional data directory.")
    parser.add_argument("--data_files", default=None, help="Optional data files (comma separated).")
    parser.add_argument("--data_probs", default=None, help="Probabilities for multiple datasets (comma separated).")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode.")
    parser.add_argument("--num_workers", type=int, default=8, help="Workers for map operations.")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Number of samples to scan (None for all; required if streaming).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Shuffle seed (if applicable).")
    parser.add_argument(
        "--parquet_file",
        default=None,
        help="If provided, read samples directly from a local Parquet file or directory of Parquet files (bypasses HF load_dataset).",
    )
    return parser.parse_args()


def _percentiles(data: List[int], ps: List[float]) -> List[Optional[float]]:
    if not data:
        return [None for _ in ps]
    data_sorted = sorted(data)
    n = len(data_sorted)
    out = []
    for p in ps:
        k = (n - 1) * p
        f = int(k)
        c = min(f + 1, n - 1)
        if f == c:
            out.append(float(data_sorted[int(k)]))
        else:
            d0 = data_sorted[f] * (c - k)
            d1 = data_sorted[c] * (k - f)
            out.append(float(d0 + d1))
    return out


def main():
    args = _parse_args()
    color = utils.Color
    if not args.parquet_file and not args.dataset:
        raise ValueError("Either --parquet_file or --dataset must be provided.")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    print(f"Loaded tokenizer: {tokenizer.__class__.__name__} ({args.tokenizer_path})")

    if args.parquet_file:
        path = args.parquet_file
        print(f"Reading directly from parquet: {path}")
        try:
            import pyarrow.parquet as pq
        except ImportError:
            pq = None

        def parquet_paths(p):
            if os.path.isdir(p):
                files = sorted(
                    os.path.join(p, f)
                    for f in os.listdir(p)
                    if f.endswith(".parquet")
                )
                if not files:
                    raise FileNotFoundError(f"No .parquet files found in directory: {p}")
                return files
            if not os.path.exists(p):
                raise FileNotFoundError(p)
            return [p]

        files = parquet_paths(path)

        if pq is not None:
            def sample_iter():
                for file in files:
                    pf = pq.ParquetFile(file)
                    for batch in pf.iter_batches(batch_size=1024):
                        for row in batch.to_pylist():
                            yield row
        else:
            import pandas as pd
            def sample_iter():
                for file in files:
                    df = pd.read_parquet(file)
                    for _, row in df.iterrows():
                        yield row.to_dict()

        it = sample_iter()
        iterator = itertools.islice(it, args.max_samples) if args.max_samples else it
    else:
        dataset = build_dataset(
            dataset=args.dataset,
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
            data_dir=args.data_dir,
            data_files=args.data_files,
            data_probs=args.data_probs,
            streaming=args.streaming,
            dp_degree=1,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        print(dataset)
        if args.streaming and args.max_samples is None:
            raise ValueError("In streaming mode you must set --max_samples to avoid unbounded iteration.")
        it = iter(dataset)
        iterator = itertools.islice(it, args.max_samples) if args.max_samples else it

    lengths: List[int] = []
    for idx, sample in enumerate(iterator, 1):
        text = _extract_text(sample)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        lengths.append(len(token_ids))
        if idx % 1000 == 0:
            print(f"[progress] processed {idx} samples, avg_len={sum(lengths)/len(lengths):.2f}")

    if not lengths:
        print("No samples processed; check dataset arguments.")
        return

    mean_len = sum(lengths) / len(lengths)
    min_len, max_len = min(lengths), max(lengths)
    p50, p90, p95, p99 = _percentiles(lengths, [0.5, 0.9, 0.95, 0.99])

    print(f"{color.green}Processed {len(lengths)} samples (max {args.max_samples}){color.reset}")
    print(f"min/mean/max: {min_len} / {mean_len:.2f} / {max_len}")
    print(f"p50/p90/p95/p99: {p50:.1f} / {p90:.1f} / {p95:.1f} / {p99:.1f}")


if __name__ == "__main__":
    main()

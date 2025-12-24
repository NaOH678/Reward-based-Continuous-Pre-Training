#!/usr/bin/env python3
"""
Download the LM-Evaluation-Harness datasets used in the paper table so they can
be copied to an offline cluster.

Run on a machine with internet access:
    python scripts/download_lm_eval_datasets.py --cache-dir /path/to/hf_cache

Then copy the cache directory to the offline machine and set HF_HOME (or
HF_DATASETS_CACHE/TRANSFORMERS_CACHE) to that location before running
lm-evaluation-harness.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Sequence

from datasets import (
    DownloadConfig,
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset,
)


@dataclass
class DatasetSpec:
    task: str
    path: str
    name: Optional[str]
    splits: Sequence[str]
    all_configs: bool = False  # If True, download every config name for this path.


DATASETS = [
    # Generative code / math
    DatasetSpec("humaneval_plus", "evalplus/humanevalplus", None, ["test"]),
    DatasetSpec("mbpp_plus", "evalplus/mbppplus", None, ["test"]),
    DatasetSpec("gsm8k", "gsm8k", None, ["train", "test"]),
    # Multi-choice general knowledge / common sense
    DatasetSpec("mmlu", "cais/mmlu", None, ["dev", "test"], all_configs=True),
    DatasetSpec("commonsense_qa", "tau/commonsense_qa", None, ["train", "validation"]),
    DatasetSpec("hellaswag", "Rowan/hellaswag", None, ["train", "validation"]),
    DatasetSpec("winogrande", "winogrande", "winogrande_xl", ["train", "validation", "test"]),
    DatasetSpec("openbookqa", "openbookqa", "main", ["train", "validation", "test"]),
    DatasetSpec("arc_easy", "allenai/ai2_arc", "ARC-Easy", ["train", "validation", "test"]),
    DatasetSpec("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", ["train", "validation", "test"]),
    DatasetSpec("piqa", "baber/piqa", None, ["train", "validation", "test"]),
    # Multi-choice text understanding
    DatasetSpec("race_high", "EleutherAI/race", "high", ["train", "validation", "test"]),
    DatasetSpec("race_middle", "EleutherAI/race", "middle", ["train", "validation", "test"]),
    DatasetSpec("boolq", "super_glue", "boolq", ["train", "validation"]),
    # Culture
    DatasetSpec("nq_open", "nq_open", None, ["train", "validation"]),
    DatasetSpec("triviaqa", "trivia_qa", "rc.nocontext", ["train", "validation"]),
]


def prepare_cache_dirs(cache_dir: str) -> str:
    """Ensure cache directories exist and set HF cache env vars."""
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(cache_dir, "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_dir, "hub"))
    return cache_dir


def download_dataset_config(
    path: str,
    name: Optional[str],
    splits: Sequence[str],
    download_config: DownloadConfig,
    hf_token: Optional[str],
) -> None:
    label = name or "default"
    print(f"* {path} ({label})")
    try:
        available = set(get_dataset_split_names(path, config_name=name, token=hf_token))
    except Exception as err:  # noqa: BLE001
        available = None
        print(f"  ! could not list splits ({err}); will try requested splits")

    for split in splits:
        if available is not None and split not in available:
            print(f"  - skip split '{split}' (not offered by this dataset)")
            continue
        print(f"  - downloading split '{split}'")
        load_dataset(
            path,
            name=name,
            split=split,
            cache_dir=download_config.cache_dir,
            download_config=download_config,
        )


def download_spec(spec: DatasetSpec, download_config: DownloadConfig, hf_token: Optional[str]) -> None:
    if spec.all_configs:
        try:
            config_names = get_dataset_config_names(spec.path, token=hf_token)
        except Exception as err:  # noqa: BLE001
            print(f"! Failed to list configs for {spec.path}: {err}")
            return
        for config_name in config_names:
            download_dataset_config(spec.path, config_name, spec.splits, download_config, hf_token)
    else:
        download_dataset_config(spec.path, spec.name, spec.splits, download_config, hf_token)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download LM-Eval datasets for offline use (matching the paper table)."
    )
    parser.add_argument(
        "--cache-dir",
        default="./hf_cache",
        help="Where to store the HF cache to copy to the offline cluster.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional HF token if you need it for gated datasets.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Subset of task names to download (defaults to all). "
        "Use names from the DATASETS list, e.g., humaneval_plus gsm8k mmlu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = prepare_cache_dirs(args.cache_dir)
    download_config = DownloadConfig(cache_dir=cache_dir, use_auth_token=args.hf_token)

    selected = DATASETS
    if args.tasks:
        wanted = set(args.tasks)
        selected = [spec for spec in DATASETS if spec.task in wanted]
        missing = wanted.difference({spec.task for spec in DATASETS})
        if missing:
            print(f"! Unknown task names (will skip): {', '.join(sorted(missing))}")

    for spec in selected:
        download_spec(spec, download_config, args.hf_token)

    print(f"\nDone. Copy '{cache_dir}' to the offline machine and set HF_HOME to that path before eval.")


if __name__ == "__main__":
    main()

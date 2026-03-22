#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from datasets import Dataset


def _read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Load a list of samples from either:
    - a single JSON object
    - a JSON array
    - a JSONL file (one JSON object per line)
    """
    text = path.read_text(encoding="utf-8").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None

    if data is not None:
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
        raise ValueError(f"Unsupported JSON root type: {type(data)}")

    samples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def _normalize_domain(raw_domain: Any) -> List[str]:
    if raw_domain is None:
        return []
    if isinstance(raw_domain, list):
        return [str(item) for item in raw_domain]
    return [str(raw_domain)]


def _to_text(sample: Dict[str, Any]) -> str:
    domain = _normalize_domain(sample.get("domain"))
    parts: List[str] = []
    if domain:
        parts.append(f"domain={' | '.join(domain)}")
    if sample.get("difficulty") is not None:
        parts.append(f"difficulty={sample['difficulty']}")
    if sample.get("source"):
        parts.append(f"source={sample['source']}")

    # Only emit problem/solution for the training text (no answer).
    problem = sample.get("problem")
    solution = sample.get("solution")
    if problem:
        parts.append(f"Human: {problem}")
    if solution:
        parts.append(f"Assistant: {solution}")

    return "\n\n".join(parts)


def convert(input_path: Path, output_path: Path) -> None:
    raw_samples = _read_json_or_jsonl(input_path)
    rows: List[Dict[str, Any]] = []
    for sample in raw_samples:
        row = dict(sample)
        row["domain"] = _normalize_domain(sample.get("domain"))
        row["text"] = _to_text(sample)
        rows.append(row)
    dataset = Dataset.from_list(rows)
    dataset.to_parquet(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert math QA JSON/JSONL into Parquet with a 'text' column for sample-level loading."
    )
    parser.add_argument("--input", required=True, help="Input JSON/JSONL file with fields like problem/solution.")
    parser.add_argument("--output", required=True, help="Output Parquet file path.")
    args = parser.parse_args()

    convert(Path(args.input), Path(args.output))
    print(f"Wrote Parquet to {args.output}")
    print(
        "Load with: build_dataset(dataset='parquet', data_files='{output}', dataset_split='train') "
        "and set --training.sample_level when training."
        .format(output=args.output)
    )


if __name__ == "__main__":
    main()

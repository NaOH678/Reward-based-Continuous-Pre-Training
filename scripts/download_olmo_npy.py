#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import concurrent.futures
import shutil
import sys
import threading
import urllib.request
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse


def extract_data_paths(config_path: Path) -> List[str]:
    urls: List[str] = []
    in_data = False
    in_paths = False
    data_indent: Optional[int] = None
    paths_indent: Optional[int] = None

    with config_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(line) - len(line.lstrip(" "))

            if not in_data:
                if stripped.startswith("data:"):
                    in_data = True
                    data_indent = indent
                continue

            if data_indent is not None and indent <= data_indent and not stripped.startswith("data:"):
                in_data = False
                in_paths = False
                data_indent = None
                paths_indent = None
                continue

            if not in_paths:
                if stripped.startswith("paths:"):
                    in_paths = True
                    paths_indent = indent
                continue

            if paths_indent is not None and indent <= paths_indent and not stripped.startswith("-"):
                in_paths = False
                paths_indent = None
                if stripped.startswith("paths:"):
                    in_paths = True
                    paths_indent = indent
                continue

            if stripped.startswith("-"):
                item = stripped[1:].strip()
                if not item or item.startswith("#"):
                    continue
                if " #" in item:
                    item = item.split(" #", 1)[0].strip()
                urls.append(item)

    return urls


def url_to_dest(url: str, output_dir: Path, flat: bool) -> Path:
    if flat:
        return output_dir / Path(urlparse(url).path).name
    rel = urlparse(url).path.lstrip("/")
    return output_dir / rel


def download_url(
    url: str,
    dest: Path,
    overwrite: bool,
    timeout: int,
    print_lock: Optional[threading.Lock] = None,
) -> None:
    if dest.exists() and not overwrite:
        if print_lock:
            with print_lock:
                print(f"skip: {dest}")
        else:
            print(f"skip: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".partial")
    req = urllib.request.Request(url, headers={"User-Agent": "flame-downloader"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp_path.open("wb") as f:
        shutil.copyfileobj(resp, f)
    tmp_path.replace(dest)
    if print_lock:
        with print_lock:
            print(f"saved: {dest}")
    else:
        print(f"saved: {dest}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download OLMo memmap .npy files referenced in a config YAML."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to OLMo YAML config (e.g., OLMo/configs/official-0425/OLMo2-1B-stage2-seed42069.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to store downloaded files.",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Store files directly in output-dir without recreating URL paths.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of files to download (for quick testing).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent download workers.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Download timeout per file in seconds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved URLs and exit without downloading.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue downloading even if a file fails.",
    )
    args = parser.parse_args()

    if not args.config.is_file():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 2

    urls = extract_data_paths(args.config)
    if args.max_files is not None:
        urls = urls[: args.max_files]

    if not urls:
        print("No data.paths URLs found in config.", file=sys.stderr)
        return 1

    if args.dry_run:
        for url in urls:
            print(url)
        return 0

    print_lock = threading.Lock()
    if args.workers <= 1:
        for url in urls:
            dest = url_to_dest(url, args.output_dir, args.flat)
            try:
                download_url(url, dest, args.overwrite, args.timeout, print_lock)
            except Exception as exc:
                if args.continue_on_error:
                    print(f"error: {url} -> {exc}", file=sys.stderr)
                    continue
                raise
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    download_url,
                    url,
                    url_to_dest(url, args.output_dir, args.flat),
                    args.overwrite,
                    args.timeout,
                    print_lock,
                ): url
                for url in urls
            }
            for future in concurrent.futures.as_completed(future_map):
                url = future_map[future]
                try:
                    future.result()
                except Exception as exc:
                    if args.continue_on_error:
                        print(f"error: {url} -> {exc}", file=sys.stderr)
                        continue
                    raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

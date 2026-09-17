#!/usr/bin/env python3
"""
Standardized Dataset Preparation for Multi-Pass UMPIRE Benchmarking (ADR 0006).

Ingests data/canonical_manifest_all.json and generates standardized question.jsonl
files for all 7 Core VLM benchmarks and 2 Robustness benchmarks:
- ai2d (N=2,000)
- chartqa (N=2,000)
- docvqa (N=2,000)
- scienceqa (N=2,000)
- textvqa (N=2,000)
- vizwiz-vqa (N=2,000)
- vqav2 (N=2,000)
- avqa (N=2,000)
- vllm-safety (N=1,900)
Total universe: 17,900 questions per model with exact 1-to-1 question ID parity.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_all_datasets")


def prepare_dataset_jsonl(
    dataset_name: str,
    samples: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Prepares a standardized question.jsonl file for a given benchmark."""
    ds_dir = output_dir / dataset_name
    ds_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = ds_dir / "questions.jsonl"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in samples:
            line_dict = {
                "question_id": item["question_id"],
                "image": item.get("image", f"{item['question_id']}.jpg"),
                "text": item.get("question", ""),
                "category": dataset_name,
                "answers": item.get("answers", []),
            }
            f.write(json.dumps(line_dict, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(samples)} samples for '{dataset_name}' -> {jsonl_path}")
    return jsonl_path


def run_preparation(manifest_path: Path, output_dir: Path) -> dict[str, int]:
    """Reads manifest and prepares questions.jsonl for all benchmarks."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    logger.info(f"Loading canonical manifest from {manifest_path}...")
    with open(manifest_path, encoding="utf-8") as f:
        manifest: dict[str, list[dict[str, Any]]] = json.load(f)

    stats: dict[str, int] = {}
    for dataset_name, samples in manifest.items():
        prepare_dataset_jsonl(dataset_name, samples, output_dir)
        stats[dataset_name] = len(samples)

    total_samples = sum(stats.values())
    logger.info(
        f"Successfully prepared all {len(stats)} datasets ({total_samples} questions total) into {output_dir}"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare All Datasets for UMPIRE Evaluation")
    parser.add_argument(
        "--manifest",
        type=str,
        default="../trajectory_calibration/data/canonical_manifest_all.json",
        help="Path to data/canonical_manifest_all.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/all_datasets",
        help="Root directory to write prepared dataset folders",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    manifest_p = Path(args.manifest)
    if not manifest_p.is_absolute():
        manifest_p = (repo_root / manifest_p).resolve()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()

    run_preparation(manifest_p, out_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Aggregate Multi-Pass UMPIRE Evaluation Results across Datasets and Rollout Budgets K.

Consolidates umpire_results_all_k.json files into summary CSV and JSON matrices.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("aggregate_umpire_results")


def parse_eval_results(eval_dir: Path) -> list[dict[str, Any]]:
    """Crawls evaluation directories and collects metric records."""
    rows: list[dict[str, Any]] = []

    pattern = str(eval_dir / "**" / "umpire_results_all_k.json")
    json_files = glob.glob(pattern, recursive=True)

    for jf_str in json_files:
        jf = Path(jf_str)
        # Directory structure: eval_dir / {arch} / {dataset} / umpire_results_all_k.json
        parts = jf.relative_to(eval_dir).parts
        if len(parts) >= 2:
            arch = parts[0]
            dataset = parts[1]
        else:
            arch = "unknown"
            dataset = jf.parent.name

        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read {jf}: {e}")
            continue

        for k_key, methods in data.items():
            k_val = int(k_key.replace("k_", "")) if "k_" in k_key else k_key
            for method, metrics in methods.items():
                row = {
                    "architecture": arch,
                    "dataset": dataset,
                    "rollout_budget_k": k_val,
                    "method": method,
                    **metrics,
                }
                rows.append(row)

    return rows


def aggregate_and_save(rows: list[dict[str, Any]], output_dir: Path) -> pd.DataFrame:
    """Saves aggregated metrics to CSV and JSON formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    if df.empty:
        logger.warning("No evaluation results found to aggregate.")
        return df

    # Reorder columns
    lead_cols = ["architecture", "dataset", "rollout_budget_k", "method"]
    metric_cols = [c for c in df.columns if c not in lead_cols]
    df = df[lead_cols + metric_cols]

    csv_path = output_dir / "umpire_summary_all_k.csv"
    json_path = output_dir / "umpire_summary_all_k.json"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=4)

    logger.info(f"Saved {len(df)} summary records to {csv_path} and {json_path}")

    # Compute macro-averages per (arch, method, K)
    if "cece" in df.columns and "auc" in df.columns:
        macro = df.groupby(["architecture", "method", "rollout_budget_k"])[["cece", "auc"]].mean().reset_index()
        macro_csv = output_dir / "umpire_macro_averages.csv"
        macro.to_csv(macro_csv, index=False)
        logger.info(f"Saved macro averages across benchmarks to {macro_csv}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate UMPIRE Multi-K Evaluation Results")
    parser.add_argument("--eval_dir", type=str, default="output/evaluation", help="Root directory containing evaluation outputs")
    parser.add_argument("--output_dir", type=str, default="output/summary", help="Directory to save summary tables")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    eval_p = Path(args.eval_dir)
    if not eval_p.is_absolute():
        eval_p = repo_root / eval_p

    out_p = Path(args.output_dir)
    if not out_p.is_absolute():
        out_p = repo_root / out_p

    rows = parse_eval_results(eval_p)
    aggregate_and_save(rows, out_p)


if __name__ == "__main__":
    main()

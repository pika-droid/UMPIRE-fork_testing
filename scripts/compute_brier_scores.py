#!/usr/bin/env python3
"""Calculate Brier score from image_df_with_uncertainty.pkl."""

import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def compute_brier(df: pd.DataFrame, random_seed: int = 10, calibration_ratio: float = 0.05) -> dict:
    """Computes calibrated Brier score for each uncertainty method."""
    # Determine ground-truth correctness
    if "is_correct" in df.columns:
        Y = df["is_correct"].astype(int).to_numpy()
    elif "exact_match" in df.columns:
        Y = (df["exact_match"] == 1).astype(int).to_numpy()
    elif "rougeL_to_target" in df.columns:
        Y = (df["rougeL_to_target"] >= 0.8).astype(int).to_numpy()
    else:
        raise ValueError("Could not find correctness column (is_correct, exact_match, or rougeL_to_target).")

    # 5% dev split for calibration (matching paper ECE protocol)
    n = len(df)
    dev_size = max(1, int(n * calibration_ratio))
    df_shuffled = df.sample(n=n, random_state=random_seed).reset_index(drop=True)
    Y_shuffled = df_shuffled["is_correct"].astype(int).to_numpy() if "is_correct" in df_shuffled.columns else (
        (df_shuffled["exact_match"] == 1).astype(int).to_numpy() if "exact_match" in df_shuffled.columns else
        (df_shuffled["rougeL_to_target"] >= 0.8).astype(int).to_numpy()
    )

    dev_df = df_shuffled.iloc[:dev_size]
    test_df = df_shuffled.iloc[dev_size:]
    Y_dev = Y_shuffled[:dev_size]
    Y_test = Y_shuffled[dev_size:]

    methods = ["umpire", "semantic_entropy", "eigen_score", "ln_entropy"]
    brier_scores = {}

    for m in methods:
        if m not in df.columns:
            continue
        try:
            lr = LogisticRegression(random_state=random_seed)
            lr.fit(dev_df[[m]].to_numpy(), Y_dev)
            probs = lr.predict_proba(test_df[[m]].to_numpy())[:, 1]
            brier_scores[m] = round(float(brier_score_loss(Y_test, probs)), 4)
        except Exception:
            brier_scores[m] = None

    return brier_scores


def process_file(pkl_path: Path) -> dict:
    """Processes a single file, prints results, and saves brier_scores.json."""
    with open(pkl_path, "rb") as f:
        df = pickle.load(f)

    scores = compute_brier(df)
    dataset_name = pkl_path.parent.name

    print(f"\nDataset: {dataset_name} ({len(df)} samples)")
    print("-" * 35)
    for method, score in scores.items():
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"{method:<20} : {score_str}")
    print("-" * 35)

    out_file = pkl_path.parent / "brier_scores.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=4)
    print(f"Saved: {out_file}\n")
    return scores


def main():
    parser = argparse.ArgumentParser(description="Calculate Brier score from image_df_with_uncertainty.pkl")
    parser.add_argument("--input_file", "-i", type=str, default=None, help="Path to image_df_with_uncertainty.pkl")
    parser.add_argument("--eval_dir", "-d", type=str, default=None, help="Directory to process all image_df_with_uncertainty.pkl files")
    args = parser.parse_args()

    if not args.input_file and not args.eval_dir:
        parser.error("Please provide --input_file or --eval_dir")

    if args.input_file:
        process_file(Path(args.input_file).resolve())
    elif args.eval_dir:
        pkl_files = list(Path(args.eval_dir).resolve().glob("**/image_df_with_uncertainty.pkl"))
        if not pkl_files:
            print(f"No image_df_with_uncertainty.pkl files found in {args.eval_dir}")
            return
        all_results = {}
        for pf in pkl_files:
            ds_name = f"{pf.parent.parent.name}/{pf.parent.name}"
            all_results[ds_name] = process_file(pf)
        summary_file = Path(args.eval_dir).resolve() / "brier_scores_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4)
        print(f"Saved summary to {summary_file}")


if __name__ == "__main__":
    main()

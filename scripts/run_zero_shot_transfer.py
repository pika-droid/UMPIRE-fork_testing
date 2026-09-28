#!/usr/bin/env python3
"""
Zero-Shot Cross-Dataset Transfer Calibration (Train on One, Test on the Rest).

Evaluates calibration transferability across different multimodal formats at K=50 rollouts:
For each source dataset D_train:
    Fit logistic calibrator on D_train for each method (umpire, ln_entropy, semantic_entropy, eigen_score).
For each target dataset D_test != D_train:
    Evaluate frozen calibrator to compute:
    1. Zero-Shot ECE (Standard Expected Calibration Error)
    2. Zero-Shot ACE (Adaptive Calibration Error / Adaptive ECE)

Produces N x N transfer matrices and summary tables for both m3 and mqt architectures.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Add repo root and modules to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))
sys.path.append(str(repo_root / "modules"))

import uncertainty_metrics.numpy as um
from modules.eval_utils import split_balanced_data, df_to_markdown_bold

METHODS = ["ln_entropy", "semantic_entropy", "eigen_score", "umpire"]
DEFAULT_DATASETS = [
    "ai2d",
    "scienceqa",
    "textvqa",
    "vizwiz-vqa",
    "vqav2",
    "avqa",
    "vllm-safety",
    "chartqa",
    "docvqa",
]


def load_dataset_scores(arch: str, dataset: str, repo_root: Path) -> pd.DataFrame:
    eval_dir = repo_root / "output" / "evaluation" / arch / dataset
    cache_file = eval_dir / "image_df_with_uncertainty.pkl"

    if cache_file.exists():
        df = pd.read_pickle(cache_file)
    else:
        # Fallback to computing scores from generations.pkl if cache not present
        gen_file = repo_root / "output" / "generations" / arch / dataset / "generations.pkl"
        if not gen_file.exists():
            return None
        print(f"Cache not found for {arch}/{dataset}, evaluating K=50 from {gen_file.name}...")
        import pickle
        from modules.logdet_utils import (
            normalize_embedding, get_quad_entropy,
            get_normalized_entropy, compute_eigenscore, slice_rollouts
        )
        from pipeline.compute_umpire_and_evaluate import (
            compute_umpire, get_logdet_term, get_adaptive_alpha_dev_set,
            compute_semantic_entropy_from_scratch, compute_semantic_entropy_from_cluster_ids
        )
        with open(gen_file, "rb") as f:
            raw_data = pickle.load(f)
        data = [slice_rollouts(s, 50) for s in raw_data]
        df = pd.DataFrame().from_dict(data)

        if 'internal_embedding' in df.columns and 'embedding' not in df.columns:
            df = df.rename(columns={'internal_embedding': 'embedding'})
        df['norm_embedding'] = df['embedding'].apply(normalize_embedding)
        df['logdet'] = df.apply(lambda x: get_logdet_term(x, jitter=1e-6), axis=1)
        df['quad_entropy'] = df['generations_log_likelihood'].apply(lambda llh: get_quad_entropy(llh))
        alpha = get_adaptive_alpha_dev_set(df, 'logdet', 'quad_entropy', dev_ratio=0.1, random_seed=10)
        df['umpire'] = df.apply(lambda x: compute_umpire(x, alpha=alpha, jitter=1e-6), axis=1)
        df['ln_entropy'] = df['generations_log_likelihood'].apply(get_normalized_entropy)
        df['eigen_score'] = df.apply(lambda x: compute_eigenscore(x, jitter=1e-6), axis=1)

        # Check cluster_cache
        cluster_cache_file = eval_dir / "cluster_cache.pkl"
        if cluster_cache_file.exists():
            with open(cluster_cache_file, "rb") as f:
                cids = pickle.load(f)
            df['cluster_ids'] = cids
            df['semantic_entropy'] = df.apply(compute_semantic_entropy_from_cluster_ids, axis=1)
        elif 'cluster_ids' in df.columns:
            df['semantic_entropy'] = df.apply(compute_semantic_entropy_from_cluster_ids, axis=1)
        else:
            from modules.semantic_entropy import EntailmentDeberta
            model = EntailmentDeberta()
            df['semantic_entropy'] = df.apply(lambda x: compute_semantic_entropy_from_scratch(x, model), axis=1)

    # Standardize ground-truth correctness label
    if 'is_correct' not in df.columns:
        if 'exact_match' in df.columns:
            df['is_correct'] = (df['exact_match'] == 1)
        elif 'rougeL_to_target' in df.columns:
            df['is_correct'] = (df['rougeL_to_target'] >= 0.8)
        else:
            raise ValueError(f"No ground truth accuracy column found in {arch}/{dataset}")

    return df


def evaluate_transfer_for_arch(arch: str, datasets: List[str], calibration_ratio: float,
                               use_full_train: bool, num_bins: int,
                               output_dir: Path, repo_root: Path):
    print(f"\n{'='*75}")
    print(f"RUNNING ZERO-SHOT CROSS-DATASET CALIBRATION TRANSFER: {arch.upper()}")
    print(f"{'='*75}")

    data_map: Dict[str, pd.DataFrame] = {}
    for ds in datasets:
        df = load_dataset_scores(arch, ds, repo_root)
        if df is not None and not df.empty:
            data_map[ds] = df
            print(f"  [+] Loaded {ds:15}: {len(df)} samples")
        else:
            print(f"  [-] Missing {ds:15} (skipped)")

    available_ds = list(data_map.keys())
    if len(available_ds) < 2:
        print(f"[!] Need at least 2 datasets to run zero-shot transfer for {arch}. Available: {available_ds}")
        return

    summary_rows = []
    all_ece_matrices = {}
    all_ace_matrices = {}

    for method in METHODS:
        ece_matrix = pd.DataFrame(index=available_ds, columns=available_ds, dtype=float)
        ace_matrix = pd.DataFrame(index=available_ds, columns=available_ds, dtype=float)

        in_domain_ece_list = []
        in_domain_ace_list = []
        zero_shot_ece_list = []
        zero_shot_ace_list = []

        for train_ds in available_ds:
            train_df = data_map[train_ds]

            # Fit logistic calibrator on source dataset
            if use_full_train:
                X_train = train_df[[method]].to_numpy()
                Y_train = train_df['is_correct'].astype(int).to_numpy()
            else:
                dev_df, _ = split_balanced_data(train_df, calibration_ratio, random_seed=10, balanced=False)
                X_train = dev_df[[method]].to_numpy()
                Y_train = dev_df['is_correct'].astype(int).to_numpy()

            # Guard against single-class in dev split
            if len(np.unique(Y_train)) < 2:
                model = LogisticRegression(random_state=10)
                # Fallback to full train set if dev set lacked variation
                X_train = train_df[[method]].to_numpy()
                Y_train = train_df['is_correct'].astype(int).to_numpy()
            else:
                model = LogisticRegression(random_state=10)

            model.fit(X_train, Y_train)

            for test_ds in available_ds:
                test_df = data_map[test_ds]
                X_test = test_df[[method]].to_numpy().reshape(-1, 1)
                Y_test = test_df['is_correct'].astype(int).to_numpy()

                p_hat = model.predict_proba(X_test)[:, 1]
                ece_val = um.ece(probs=p_hat, labels=Y_test, num_bins=num_bins)
                ace_val = um.ace(probs=p_hat, labels=Y_test, num_bins=num_bins)

                ece_matrix.loc[train_ds, test_ds] = ece_val
                ace_matrix.loc[train_ds, test_ds] = ace_val

                if train_ds == test_ds:
                    in_domain_ece_list.append(ece_val)
                    in_domain_ace_list.append(ace_val)
                else:
                    zero_shot_ece_list.append(ece_val)
                    zero_shot_ace_list.append(ace_val)

        # Record summary
        mean_id_ece = float(np.mean(in_domain_ece_list)) if in_domain_ece_list else np.nan
        mean_zs_ece = float(np.mean(zero_shot_ece_list)) if zero_shot_ece_list else np.nan
        mean_id_ace = float(np.mean(in_domain_ace_list)) if in_domain_ace_list else np.nan
        mean_zs_ace = float(np.mean(zero_shot_ace_list)) if zero_shot_ace_list else np.nan

        summary_rows.append({
            "architecture": arch,
            "method": method,
            "in_domain_ece": mean_id_ece,
            "zero_shot_transfer_ece": mean_zs_ece,
            "in_domain_ace": mean_id_ace,
            "zero_shot_transfer_ace": mean_zs_ace,
        })

        all_ece_matrices[method] = ece_matrix
        all_ace_matrices[method] = ace_matrix

        # Save per-method matrix CSVs
        method_ece_path = output_dir / f"zero_shot_{arch}_{method}_ece_matrix.csv"
        method_ace_path = output_dir / f"zero_shot_{arch}_{method}_ace_matrix.csv"
        ece_matrix.to_csv(method_ece_path)
        ace_matrix.to_csv(method_ace_path)

    # Save summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / f"zero_shot_{arch}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"\n--- {arch.upper()} ZERO-SHOT CALIBRATION TRANSFER SUMMARY ---")
    display_df = summary_df.copy()
    for col in display_df.columns:
        if display_df[col].dtype in [float, int]:
            display_df[col] = display_df[col].round(3)
    print(display_df.to_markdown(index=False))
    print(f"\nAll transfer matrices saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Cross-Dataset Calibration Transfer")
    parser.add_argument("--arch", type=str, default="all", choices=["m3", "mqt", "all"],
                        help="Model architecture ('m3', 'mqt', or 'all')")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names (default: all 9 benchmarks)")
    parser.add_argument("--output_dir", type=str, default="output/summary",
                        help="Directory to save summary tables")
    parser.add_argument("--calibration_ratio", type=float, default=0.05,
                        help="Calibration split ratio on training dataset (default: 0.05)")
    parser.add_argument("--use_full_train", action="store_true",
                        help="Fit calibrator on 100%% of source dataset rather than 5%% dev split")
    parser.add_argument("--num_bins", type=int, default=50,
                        help="Number of bins for ECE and ACE calculation (default: 50)")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        datasets = DEFAULT_DATASETS

    archs = ["m3", "mqt"] if args.arch == "all" else [args.arch]

    for arch in archs:
        evaluate_transfer_for_arch(
            arch=arch,
            datasets=datasets,
            calibration_ratio=args.calibration_ratio,
            use_full_train=args.use_full_train,
            num_bins=args.num_bins,
            output_dir=output_dir,
            repo_root=repo_root,
        )


if __name__ == "__main__":
    main()

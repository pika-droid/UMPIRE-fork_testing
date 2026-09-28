#!/usr/bin/env python3
"""
Per-Dataset Evaluation CLI for UMPIRE Multi-K Evaluation & Cluster Caching.

Allows running single datasets or lists of datasets independently in parallel
across multiple terminals, GPUs, or machines for both m3 and mqt architectures.

Examples:
    # Terminal 1 on GPU 0 (or Machine 1):
    python scripts/run_dataset_evaluation.py --arch m3 --dataset ai2d

    # Terminal 2 on GPU 1 (or Machine 2):
    python scripts/run_dataset_evaluation.py --arch mqt --dataset scienceqa

    # Run a batch of datasets on one machine:
    python scripts/run_dataset_evaluation.py --arch m3 --datasets vqav2,textvqa,docvqa
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

ALL_DATASETS = [
    "ai2d",
    "chartqa",
    "docvqa",
    "scienceqa",
    "textvqa",
    "vizwiz-vqa",
    "vqav2",
    "avqa",
    "vllm-safety",
]


def evaluate_dataset(arch: str, dataset: str, budgets: str, jitter: float,
                     calibration_model: str, skip_existing: bool,
                     force_recluster: bool, repo_root: Path) -> bool:
    gen_file = repo_root / "output" / "generations" / arch / dataset / "generations.pkl"
    eval_out = repo_root / "output" / "evaluation" / arch / dataset
    eval_out.mkdir(parents=True, exist_ok=True)

    if not gen_file.exists():
        print(f"[-] SKIPPING {arch}/{dataset}: Generation file not found at {gen_file}", flush=True)
        return False

    all_k_file = eval_out / "umpire_results_all_k.json"
    cache_df_file = eval_out / "image_df_with_uncertainty.pkl"

    if skip_existing and all_k_file.exists() and cache_df_file.exists():
        print(f"[+] SKIPPING {arch}/{dataset}: Already evaluated and cached.", flush=True)
        return True

    eval_script = repo_root / "pipeline" / "compute_umpire_and_evaluate.py"
    cmd = [
        sys.executable,
        str(eval_script),
        "--generation_file", str(gen_file),
        "--output_dir", str(eval_out),
        "--jitter", str(jitter),
        "--calibration_model", calibration_model,
        "--rollout_budgets", budgets,
    ]
    if force_recluster:
        cmd.append("--re_cluster_semantic_entropy")

    print(f"\n{'='*75}", flush=True)
    print(f"--> LAUNCHING EVALUATION: Architecture: {arch.upper()} | Dataset: {dataset}", flush=True)
    print(f"    Command: {' '.join(cmd)}", flush=True)
    print(f"{'='*75}\n", flush=True)

    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(repo_root))
    elapsed = time.time() - t0

    if res.returncode == 0:
        print(f"\n[+] COMPLETED {arch}/{dataset} in {elapsed:.2f}s ({elapsed/60:.2f}m).", flush=True)
        print(f"    Results saved to: {eval_out}\n", flush=True)
        return True
    else:
        print(f"\n[!] ERROR: Evaluation failed for {arch}/{dataset} with exit code {res.returncode}\n", flush=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run UMPIRE Evaluation Per Dataset")
    parser.add_argument("--arch", type=str, required=True, choices=["m3", "mqt"],
                        help="Model architecture ('m3' or 'mqt')")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Single dataset name or 'all'")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names (e.g. 'ai2d,scienceqa,vqav2')")
    parser.add_argument("--budgets", type=str, default="10,20,30,40,50",
                        help="Comma-separated rollout budgets K (default: '10,20,30,40,50')")
    parser.add_argument("--jitter", type=float, default=1e-6,
                        help="Jitter value for logdet numerical stability (default: 1e-6)")
    parser.add_argument("--calibration_model", type=str, default="logistic",
                        choices=["logistic", "minmax", "isotonic", "linear"],
                        help="Calibration model type (default: logistic)")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip dataset if umpire_results_all_k.json and cache exist (default: True)")
    parser.add_argument("--force_recompute", action="store_true",
                        help="Force re-evaluation even if outputs exist")
    parser.add_argument("--force_recluster", action="store_true",
                        help="Force re-clustering with DeBERTa from scratch")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    # Determine datasets to run
    target_datasets = []
    if args.dataset:
        if args.dataset.lower() == "all":
            target_datasets = ALL_DATASETS
        else:
            target_datasets = [args.dataset.strip()]
    elif args.datasets:
        target_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        print("Error: Specify either --dataset <name|all> or --datasets <list,of,datasets>.")
        sys.exit(1)

    skip_existing = args.skip_existing and not args.force_recompute

    print(f"Running evaluation for {args.arch.upper()} on {len(target_datasets)} dataset(s): {target_datasets}")
    successes = []
    failures = []

    for ds in target_datasets:
        ok = evaluate_dataset(
            arch=args.arch,
            dataset=ds,
            budgets=args.budgets,
            jitter=args.jitter,
            calibration_model=args.calibration_model,
            skip_existing=skip_existing,
            force_recluster=args.force_recluster,
            repo_root=repo_root,
        )
        if ok:
            successes.append(ds)
        else:
            failures.append(ds)

    print(f"\n{'='*75}")
    print(f"BATCH RUN FINISHED: {len(successes)} succeeded, {len(failures)} failed.")
    if successes:
        print(f"  Succeeded: {', '.join(successes)}")
    if failures:
        print(f"  Failed:    {', '.join(failures)}")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()

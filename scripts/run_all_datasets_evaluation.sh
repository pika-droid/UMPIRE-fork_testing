#!/usr/bin/env bash
# ==============================================================================
# RunPod Script: Multi-K Rollout Slicing Evaluation Across All 9 Benchmarks
# ==============================================================================
set -euo pipefail

REPO_ROOT="/workspace/umpire_testing"
cd "${REPO_ROOT}"

DATASETS=(
    "ai2d"
    "chartqa"
    "docvqa"
    "scienceqa"
    "textvqa"
    "vizwiz-vqa"
    "vqav2"
    "avqa"
    "vllm-safety"
)

ARCHS=("m3" "mqt")

echo "=== Executing Multi-K Evaluation (K in {10, 20, 30, 40, 50}) ==="
for arch in "${ARCHS[@]}"; do
    for ds in "${DATASETS[@]}"; do
        GEN_FILE="output/generations/${arch}/${ds}/generations.pkl"
        EVAL_OUT="output/evaluation/${arch}/${ds}"

        if [ ! -f "${GEN_FILE}" ]; then
            echo "Skipping ${arch}/${ds}: ${GEN_FILE} not found."
            continue
        fi

        echo "--> Evaluating ${arch} on ${ds}..."
        python pipeline/compute_umpire_and_evaluate.py \
            --generation_file "${GEN_FILE}" \
            --output_dir "${EVAL_OUT}" \
            --jitter 1e-6 \
            --calibration_model logistic \
            --rollout_budgets "10,20,30,40,50"
    done
done

echo "=== Aggregating all evaluation results ==="
python scripts/aggregate_umpire_results.py \
    --eval_dir "output/evaluation" \
    --output_dir "output/summary"

echo "================================================================="
echo "Multi-K evaluation completed successfully!"
echo "================================================================="

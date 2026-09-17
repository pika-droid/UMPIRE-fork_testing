#!/usr/bin/env bash
# ==============================================================================
# RunPod Script: Parallel Batched Multi-Rollout Generation for All 9 Datasets
# ==============================================================================
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

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

MODELS=(
    "m3:mucai/llava-v1.5-7b-m3"
    "mqt:gordonhu/MQT-LLaVA-7b"
)

# Step 1: Prepare all standardized questions.jsonl from canonical manifest
echo "=== Step 1: Preparing all datasets from canonical manifest ==="
python pipeline/prepare_all_datasets.py \
    --manifest "../trajectory_calibration/data/canonical_manifest_all.json" \
    --output_dir "data/all_datasets"

# Step 2: Execute batched generation (K=50 rollouts per sample)
echo "=== Step 2: Executing batched 50-rollout generation ==="
for model_spec in "${MODELS[@]}"; do
    ARCH="${model_spec%%:*}"
    MODEL_PATH="${model_spec#*:}"
    echo "--------------------------------------------------------"
    echo "Running Model: ${ARCH} (${MODEL_PATH})"
    echo "--------------------------------------------------------"

    for ds in "${DATASETS[@]}"; do
        Q_FILE="data/all_datasets/${ds}/questions.jsonl"
        OUT_DIR="output/generations/${ARCH}/${ds}"
        echo "--> Processing dataset: ${ds} (Arch: ${ARCH})"

        python pipeline/generate_and_compute_emb_hf.py \
            --model_path "${MODEL_PATH}" \
            --arch "${ARCH}" \
            --dataset "${ds}" \
            --question_file "${Q_FILE}" \
            --outdir "${OUT_DIR}" \
            --num_generations_per_prompt 50 \
            --temperature 1.0 \
            --top_p 0.9 \
            --max_new_tokens 32
    done
done

echo "================================================================="
echo "All 9 datasets generated successfully for both M3 and MQT!"
echo "================================================================="

import os
import sys
import subprocess
import time
import json
from pathlib import Path

datasets = [
    'docvqa',
    'scienceqa',
    'textvqa',
    'vizwiz-vqa',
    'vqav2',
    'avqa',
    'vllm-safety',
]

repo_root = Path(__file__).resolve().parent.parent
python_exe = repo_root / '.venv' / 'Scripts' / 'python.exe'
eval_script = repo_root / 'pipeline' / 'compute_umpire_and_evaluate.py'

for ds in datasets:
    gen_file = repo_root / 'output' / 'generations' / 'm3' / ds / 'generations.pkl'
    out_dir = repo_root / 'output' / 'evaluation' / 'm3' / ds
    out_dir.mkdir(parents=True, exist_ok=True)

    all_k_file = out_dir / 'umpire_results_all_k.json'

    # Check which budgets already exist
    all_budgets = [10, 20, 30, 40, 50]
    missing_budgets = []
    for b in all_budgets:
        kb_file = out_dir / f'umpire_results_k{b}.json'
        if not kb_file.exists():
            missing_budgets.append(b)

    if not missing_budgets and all_k_file.exists():
        print(f"[SKIP] {ds} is already 100% complete with all budgets.", flush=True)
        continue

    t0 = time.time()
    for b in missing_budgets:
        cmd = [
            str(python_exe),
            str(eval_script),
            '--generation_file', str(gen_file),
            '--output_dir', str(out_dir),
            '--jitter', '1e-6',
            '--calibration_model', 'logistic',
            '--rollout_budgets', str(b)
        ]

        print(f"\n==================================================", flush=True)
        print(f"STARTING EVALUATION: {ds} (Budget: K={b})", flush=True)
        print(f"Command: {' '.join(cmd)}", flush=True)
        print(f"==================================================", flush=True)

        t0 = time.time()
        res = subprocess.run(cmd, cwd=str(repo_root))
        if res.returncode != 0:
            print(f"[ERROR] Evaluation failed for {ds} K={b} with code {res.returncode}", flush=True)
            sys.exit(res.returncode)
        print(f"[COMPLETED] {ds} K={b} in {time.time() - t0:.2f} seconds.", flush=True)

    # Consolidate all_k_results across all 5 budgets
    combined_all_k = {}
    for b in all_budgets:
        kb_file = out_dir / f'umpire_results_k{b}.json'
        if kb_file.exists():
            with open(kb_file, 'r', encoding='utf-8') as f:
                combined_all_k[f'k_{b}'] = json.load(f)

    if combined_all_k:
        with open(all_k_file, 'w', encoding='utf-8') as f:
            json.dump(combined_all_k, f, indent=4)
        print(f"Consolidated {len(combined_all_k)} budgets into {all_k_file}", flush=True)

    # Ensure umpire_results.json exists (matches K=50)
    k50_file = out_dir / 'umpire_results_k50.json'
    results_file = out_dir / 'umpire_results.json'
    if k50_file.exists():
        with open(k50_file, 'r', encoding='utf-8') as f_in, open(results_file, 'w', encoding='utf-8') as f_out:
            f_out.write(f_in.read())

    print(f"[COMPLETED] {ds} in {time.time() - t0:.2f} seconds.", flush=True)

print("\n==================================================", flush=True)
print("ALL DATASETS EVALUATED SUCCESSFULLY! AGGREGATING...", flush=True)
print("==================================================", flush=True)

agg_cmd = [
    str(python_exe),
    str(repo_root / 'scripts' / 'aggregate_umpire_results.py'),
    '--eval_dir', str(repo_root / 'output' / 'evaluation'),
    '--output_dir', str(repo_root / 'output' / 'summary')
]
res = subprocess.run(agg_cmd, cwd=str(repo_root))
if res.returncode != 0:
    print(f"[ERROR] Aggregation failed with code {res.returncode}", flush=True)
    sys.exit(res.returncode)

print("\n[SUCCESS] Pipeline finished and all results aggregated.", flush=True)

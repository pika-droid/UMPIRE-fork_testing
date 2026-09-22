import os
import sys
import subprocess
import time

datasets = [
    'chartqa',
    'docvqa',
    'scienceqa',
    'textvqa',
    'vizwiz-vqa',
    'vqav2',
    'avqa',
    'vllm-safety',
]

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
python_exe = os.path.join(repo_root, '.venv', 'Scripts', 'python.exe')

for ds in datasets:
    gen_file = os.path.join(repo_root, 'output', 'generations', 'm3', ds, 'generations.pkl')
    out_dir = os.path.join(repo_root, 'output', 'evaluation', 'm3', ds)
    cmd = [
        python_exe,
        os.path.join(repo_root, 'pipeline', 'compute_umpire_and_evaluate.py'),
        '--generation_file', gen_file,
        '--output_dir', out_dir,
        '--jitter', '1e-6',
        '--calibration_model', 'logistic',
        '--rollout_budgets', '10,20,30,40,50'
    ]
    print(f"\n==================================================", flush=True)
    print(f"STARTING EVALUATION: {ds}", flush=True)
    print(f"Command: {' '.join(cmd)}", flush=True)
    print(f"==================================================", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=repo_root)
    if res.returncode != 0:
        print(f"[ERROR] Evaluation failed for {ds} with code {res.returncode}", flush=True)
        sys.exit(res.returncode)
    print(f"[COMPLETED] {ds} in {time.time() - t0:.2f} seconds.", flush=True)

print("\n==================================================", flush=True)
print("ALL DATASETS EVALUATED SUCCESSFULLY! AGGREGATING...", flush=True)
print("==================================================", flush=True)

agg_cmd = [
    python_exe,
    os.path.join(repo_root, 'scripts', 'aggregate_umpire_results.py'),
    '--eval_dir', os.path.join(repo_root, 'output', 'evaluation'),
    '--output_dir', os.path.join(repo_root, 'output', 'summary')
]
res = subprocess.run(agg_cmd, cwd=repo_root)
if res.returncode != 0:
    print(f"[ERROR] Aggregation failed with code {res.returncode}", flush=True)
    sys.exit(res.returncode)

print("\n[SUCCESS] Pipeline finished and all results aggregated.", flush=True)

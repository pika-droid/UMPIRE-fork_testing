import os
import sys
import pickle
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path

datasets = [
    'ai2d',
    'chartqa',
    'docvqa',
    'scienceqa',
    'textvqa',
    'vizwiz-vqa',
    'vqav2',
    'avqa',
    'vllm-safety'
]

def audit_dataset_data(ds_name, data, arch_name):
    total = len(data)
    expected_n = 1900 if ds_name == 'vllm-safety' else 2000
    
    rollout_lens = []
    emb_shapes = []
    has_nan_emb = 0
    has_nan_llh = 0
    empty_gens = 0
    
    em_vals = []
    rl_vals = []
    
    for s in data:
        gens = s.get('generations_text', [])
        rollout_lens.append(len(gens))
        if any(not str(g).strip() for g in gens):
            empty_gens += 1
            
        emb = s.get('internal_embedding')
        if emb is not None:
            emb_arr = np.array(emb)
            emb_shapes.append(emb_arr.shape)
            if np.isnan(emb_arr).any() or np.isinf(emb_arr).any():
                has_nan_emb += 1
                
        llhs = s.get('generations_log_likelihood', [])
        for llh in llhs:
            llh_arr = np.array(llh)
            if np.isnan(llh_arr).any() or np.isinf(llh_arr).any():
                has_nan_llh += 1
                break
                
        em_vals.append(float(s.get('exact_match', 0.0)))
        rl_vals.append(float(s.get('rougeL_to_target', 0.0)))
        
    em_sum = sum(1 for x in em_vals if x == 1.0)
    rl_ge_08 = sum(1 for x in rl_vals if x >= 0.8)
    
    return {
        'arch': arch_name,
        'dataset': ds_name,
        'count': total,
        'expected_n': expected_n,
        'count_ok': total == expected_n,
        'min_rollouts': min(rollout_lens) if rollout_lens else 0,
        'max_rollouts': max(rollout_lens) if rollout_lens else 0,
        'empty_rollout_samples': empty_gens,
        'nan_emb_samples': has_nan_emb,
        'nan_llh_samples': has_nan_llh,
        'em_count': em_sum,
        'em_pct': em_sum / total if total else 0,
        'rl_ge_0.8_count': rl_ge_08,
        'rl_ge_0.8_pct': rl_ge_08 / total if total else 0,
    }

print("=" * 80)
print("AUDITING LOCAL M3 DATASETS")
print("=" * 80)
m3_results = []
for ds in datasets:
    p = Path(f'output/generations/m3/{ds}/generations.pkl')
    if not p.exists():
        print(f"M3 {ds}: NOT FOUND")
        continue
    with open(p, 'rb') as f:
        d = pickle.load(f)
    res = audit_dataset_data(ds, d, 'm3')
    m3_results.append(res)

print("=" * 80)
print("AUDITING REMOTE MQT DATASETS (from zip archive)")
print("=" * 80)
mqt_results = []
zip_p = r'C:\Users\ashmi\Downloads\mqt_generations.zip'
if os.path.exists(zip_p):
    with zipfile.ZipFile(zip_p) as z:
        for ds in datasets:
            entry = f'mqt/{ds}/generations.pkl'
            try:
                with z.open(entry) as f:
                    d = pickle.load(f)
                res = audit_dataset_data(ds, d, 'mqt')
                mqt_results.append(res)
            except KeyError:
                print(f"MQT {ds}: NOT FOUND IN ZIP")
else:
    print(f"MQT zip not found at {zip_p}")

df_m3 = pd.DataFrame(m3_results)
df_mqt = pd.DataFrame(mqt_results)

print("\n--- M3 AUDIT SUMMARY ---")
cols_to_show = ['dataset', 'count', 'min_rollouts', 'max_rollouts', 'nan_emb_samples', 'nan_llh_samples', 'em_count', 'em_pct', 'rl_ge_0.8_count', 'rl_ge_0.8_pct']
print(df_m3[cols_to_show].to_string(index=False))

print("\n--- MQT AUDIT SUMMARY ---")
print(df_mqt[cols_to_show].to_string(index=False))

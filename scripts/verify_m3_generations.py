import os
import sys
import pickle
import numpy as np
import pandas as pd

DATASET_EXPECTATIONS = {
    'ai2d': 2000,
    'chartqa': 2000,
    'docvqa': 2000,
    'scienceqa': 2000,
    'textvqa': 2000,
    'vizwiz-vqa': 2000,
    'vqav2': 2000,
    'avqa': 2000,
    'vllm-safety': 1900,
}

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output', 'generations', 'm3')

def verify_dataset(name, expected_count):
    pkl_path = os.path.join(BASE_DIR, name, 'generations.pkl')
    if not os.path.exists(pkl_path):
        return {
            'dataset': name,
            'status': 'MISSING',
            'samples': 0,
            'expected': expected_count,
            'rollouts': 0,
            'emb_shape': 'N/A',
            'nan_inf': 'N/A',
            'error': f"File not found: {pkl_path}"
        }

    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        return {
            'dataset': name,
            'status': 'CORRUPT',
            'samples': 0,
            'expected': expected_count,
            'rollouts': 0,
            'emb_shape': 'N/A',
            'nan_inf': 'N/A',
            'error': str(e)
        }

    sample_count = len(data)
    count_ok = (sample_count == expected_count)

    rollout_ok = True
    emb_shape_ok = True
    no_nan_inf = True
    detected_shape = None
    has_cluster_ids = 0

    for idx, sample in enumerate(data):
        texts = sample.get('generations_text', [])
        llhs = sample.get('generations_log_likelihood', [])
        if len(texts) != 50 or len(llhs) != 50:
            rollout_ok = False

        emb = sample.get('embedding', sample.get('internal_embedding', None))
        if emb is None:
            emb_shape_ok = False
        else:
            if not isinstance(emb, np.ndarray):
                emb = np.array(emb)
            if idx == 0:
                detected_shape = emb.shape
            if emb.shape != (50, 4096):
                emb_shape_ok = False
            if np.isnan(emb).any() or np.isinf(emb).any():
                no_nan_inf = False

        if 'cluster_ids' in sample:
            has_cluster_ids += 1

    status = 'PASS' if (count_ok and rollout_ok and emb_shape_ok and no_nan_inf) else 'FAIL'
    return {
        'dataset': name,
        'status': status,
        'samples': sample_count,
        'expected': expected_count,
        'rollouts_per_sample': 50 if rollout_ok else 'MISMATCH',
        'emb_shape': str(detected_shape) if emb_shape_ok else 'INVALID',
        'nan_inf': 'NONE' if no_nan_inf else 'DETECTED',
        'has_precomputed_clusters': f"{has_cluster_ids}/{sample_count}",
        'error': ''
    }

def main():
    print(f"Verifying M3 datasets in: {BASE_DIR}")
    results = []
    all_passed = True
    total_samples = 0

    for name, expected_count in DATASET_EXPECTATIONS.items():
        res = verify_dataset(name, expected_count)
        results.append(res)
        total_samples += res['samples']
        if res['status'] != 'PASS':
            all_passed = False

    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("M3-LLaVA DATASET INTEGRITY VERIFICATION SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    print(f"Total samples verified: {total_samples} / 17900 expected.")

    if not all_passed or total_samples != 17900:
        print("\n[ERROR] Verification FAILED for one or more datasets!")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All datasets passed sample parity and integrity checks.")
        sys.exit(0)

if __name__ == '__main__':
    main()

import numpy as np


def normalize_embedding(x):
    return np.array([e / np.linalg.norm(e, ord=2) for e in x], dtype=np.float32)

def get_generation_embeddings(sample):
    embeddings = np.asarray(sample['norm_embedding']) # take norm_embedding instead of embedding to avoid the need to normalize here
    k = len(sample['generations_log_likelihood'])
    if embeddings.shape[0] == k + 1:
        embeddings = embeddings[1:]
    return embeddings

def get_predictive_entropy(x):
    x_ = [-np.sum(np.exp(i)*i)for i in x] 
    return np.average(x_)

def get_quad_entropy(log_likelihoods, length_normalize=False):
    if length_normalize:
        prob = np.array([np.exp(np.sum(i) / len(i)) for i in log_likelihoods])
    else:
        prob = np.array([np.exp(np.sum(i)) for i in log_likelihoods])
    incoherence_scores = 1 - prob
    return np.sum(incoherence_scores) * (1/(len(log_likelihoods)))

def get_normalized_entropy(x):
    x_ = [-np.sum(np.exp(i)*i) *(1/len(i)) for i in x]
    return np.average(x_)

def compute_logdet(K, jitter=1e-6):
    K = np.asarray(K, dtype=np.float64)
    eigvals = np.linalg.eigvalsh(0.5 * (K + K.T))
    return float(np.sum(np.log(np.maximum(eigvals, 0.0) + jitter)))

def slice_rollouts(sample, k):
    s = dict(sample)
    if 'generations_text' in s and s['generations_text'] is not None:
        s['generations_text'] = s['generations_text'][:k]
    if 'generations_log_likelihood' in s and s['generations_log_likelihood'] is not None:
        s['generations_log_likelihood'] = s['generations_log_likelihood'][:k]
    if 'norm_embedding' in s and s['norm_embedding'] is not None:
        s['norm_embedding'] = s['norm_embedding'][:k]
    if 'internal_embedding' in s and s['internal_embedding'] is not None:
        s['internal_embedding'] = s['internal_embedding'][:k]
    if 'embedding' in s and s['embedding'] is not None:
        s['embedding'] = s['embedding'][:k]
    if 'cluster_ids' in s and s['cluster_ids'] is not None:
        s['cluster_ids'] = s['cluster_ids'][:k]
    return s

# Compute the eigenvalue-based score, adapted from https://github.com/D2I-ai/eigenscore/blob/main/func/metric.py
def compute_eigenscore(row, jitter = 1e-3):
    embedding = np.asarray(get_generation_embeddings(row), dtype=np.float64)
    CovMatrix = np.cov(embedding)
    # CovMatrix = np.matmul(embedding, embedding.T)
    u, s, vT = np.linalg.svd(CovMatrix+jitter*np.eye(CovMatrix.shape[0]))
    eigenIndicator = np.mean(np.log10(s))
    return eigenIndicator
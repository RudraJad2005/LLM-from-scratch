import numpy as np

from src.math.linear_algebra import stable_softmax


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = stable_softmax(scores, axis=-1)
    return weights @ v, weights

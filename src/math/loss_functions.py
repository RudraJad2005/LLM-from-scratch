import numpy as np

def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:

    if logits.shape == targets.shape:
        max_logits = np.max(logits, axis=-1, keepdims=True)

        stable_logits = logits - max_logits
        log_probs = stable_logits - np.log(np.sum(np.exp(stable_logits), axis=-1, keepdims=True))

        return -np.sum(targets * log_probs, axis=-1)
    else:
        raise ValueError(f"Logits and targets must have the same shape, got {logits.shape} and {targets.shape}")

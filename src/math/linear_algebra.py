import numpy as np


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    check_shapes(a, b)
    return a @ b

def mat_transpose(a: np.ndarray) -> np.ndarray:
    return np.transpose(a)

def check_shapes(a: np.ndarray, b: np.ndarray) -> bool:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"matmul expects 2D matrices, got a.ndim={a.ndim}, b.ndim={b.ndim}"
        )

    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible shapes for matmul: {a.shape} and {b.shape}")
    return True


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return shifted - log_sum_exp
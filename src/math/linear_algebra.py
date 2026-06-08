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

def matrix_multiply(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    # 1. Handle empty matrices to prevent IndexError on a[0] or b[0]
    if not a or not b or not a[0] or not b[0]:
        return []
        
    # 2. Validate that the matrices can actually be multiplied
    if len(a[0]) != len(b):
        raise ValueError("Number of columns in 'a' must equal the number of rows in 'b'")

    # Create the empty result matrix
    result = [[0 for _ in range(len(b[0]))] for _ in range(len(a))]

    # Your original (and correct) multiplication logic
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                result[i][j] += a[i][k] * b[k][j]
    
    # 3. Return the result instead of just printing it
    return result

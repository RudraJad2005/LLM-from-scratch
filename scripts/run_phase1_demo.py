from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.math.linear_algebra import matmul, stable_softmax, mat_transpose, log_softmax


def main():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[2.0, 0.0], [1.0, 2.0]])
    c = matmul(a, b)
    probs = stable_softmax(c, axis=-1)
    print("Matmul result:\n", c)
    print("Softmax:\n", probs)

    e = mat_transpose(c)
    print("Transposed:\n", e)

    invalid_tensor_a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    invalid_tensor_b = np.array([[1.0, 2.0], [3.0, 4.0]])

    try:
        _ = matmul(invalid_tensor_a, invalid_tensor_b)
    except ValueError as err:
        print("Expected shape-check error:", err)

    log_probs = log_softmax(a, axis=-1)
    print("Log softmax:\n", log_probs)
    



if __name__ == "__main__":
    main()

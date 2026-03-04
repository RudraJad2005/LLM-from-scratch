import sys
from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.math.linear_algebra import log_softmax


def test_log_softmax_stability():
    x = np.array([[1000.0, 100.0], [100.0, 1000.0]])
    log_probs = log_softmax(x, axis=-1)
    assert np.all(np.isfinite(log_probs))


def test_log_softmax_probabilities_sum_to_one():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 4.5, 5.0]])
    log_probs = log_softmax(x, axis=-1)
    probs = np.exp(log_probs)
    sums = np.sum(probs, axis=-1)
    assert np.allclose(sums, 1.0, atol=1e-7)


def test_log_softmax_shift_invariance():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    shift = 10000.0
    base = log_softmax(x, axis=-1)
    shifted = log_softmax(x + shift, axis=-1)
    assert np.allclose(base, shifted, atol=1e-7)


def run_all_tests():
    test_log_softmax_stability()
    test_log_softmax_probabilities_sum_to_one()
    test_log_softmax_shift_invariance()
    print("All log_softmax tests passed.")


if __name__ == "__main__":
    run_all_tests()

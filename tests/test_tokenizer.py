import sys
from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nlp.tokenizer_bpe import train_bpe
from src.math.linear_algebra import log_softmax

def test_train_bpe_returns_merges():
    merges = train_bpe(["banana", "bandana"], num_merges=3)
    assert len(merges) <= 3


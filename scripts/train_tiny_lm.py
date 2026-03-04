from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.transformer import TinyTransformerBlock
from src.training.trainer import Trainer


def main():
    model = TinyTransformerBlock(d_model=64)
    trainer = Trainer(model)
    metrics = trainer.train_step(batch=None)
    print("Train step metrics:", metrics)


if __name__ == "__main__":
    main()

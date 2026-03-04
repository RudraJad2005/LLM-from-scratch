from dataclasses import dataclass


@dataclass
class TrainConfig:
    d_model: int = 128
    batch_size: int = 16
    learning_rate: float = 3e-4
    max_steps: int = 1000

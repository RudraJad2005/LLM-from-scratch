def cosine_lr(step: int, max_steps: int, base_lr: float) -> float:
    if max_steps <= 0:
        return base_lr
    ratio = step / max_steps
    return 0.5 * base_lr * (1 + __import__("math").cos(__import__("math").pi * ratio))

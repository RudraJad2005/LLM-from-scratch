import numpy as np

from src.model.attention import scaled_dot_product_attention


class TinyTransformerBlock:
    def __init__(self, d_model: int):
        self.d_model = d_model

    def forward(self, x: np.ndarray) -> np.ndarray:
        attn_out, _ = scaled_dot_product_attention(x, x, x)
        return x + attn_out

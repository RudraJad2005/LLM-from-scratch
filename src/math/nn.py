import random
from src.math.autodiff import Value

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.relu()
        return out

if __name__ == "__main__":
    n = Neuron(2)
    x = [Value(1.0), Value(-2.0)]
    out = n(x)

    out.backward()
    print(f"Output: {out.data}")
    print(f"Gradient of weight 1: {n.w[0].grad}")
    print(f"Gradient of weight 2: {n.w[1].grad}")

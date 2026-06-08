import random
from src.math.autodiff import Value

class Neuron:
    def __init__(self, nin):
        # nin: number of inputs
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.relu()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        # nin: number of inputs per neuron
        # nout: number of neurons in this layer
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        # nin: number of inputs
        # nouts: list of sizes of all the layers you want
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

if __name__ == "__main__":
    # Create an MLP: 3 inputs, 2 hidden layers of 4, 1 output
    model = MLP(3, [4, 4, 1])
    
    # Example input
    x = [Value(2.0), Value(3.0), Value(-1.0)]
    
    # Forward pass
    out = model(x)
    
    # Backward pass
    out.backward()
    
    print(f"MLP Output: {out.data}")
    print(f"Number of parameters: {len(model.parameters())}")

class Value:
    def __init__(self, data: float):
        self.data = data
        self.grad = 0.0

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

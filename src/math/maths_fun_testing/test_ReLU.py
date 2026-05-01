from src.math.autodiff import Value


x1 = Value(10.0)
y1 = x1.relu()
y1.backward()
print(f"Positive input: x=10, grad={x1.grad}")

x2 = Value(-5.0)
y2 = x1.relu()
y2.backward()
print(f"Negative input: x=-5, grad={x2.grad}")
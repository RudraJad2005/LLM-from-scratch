import numpy as np 
import matplotlib.pyplot as plt


# Fake data generation for regression

np.random.seed(42)

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

w = np.random.randn(1, 1)
b = 0.0
lr = 0.1

epochs = 100
n = len(X)
losses = []

# Forward pass

for epoch in range(epochs):

    y_pred = X @ w + b

    loss = np.mean((y - y_pred) ** 2)
    losses.append(loss)

    # Computeing gradients

    error = y - y_pred

    dw = (-2 / n) * (X.T @ error)
    db = (-2 / n) * np.sum(error)


    w = w - lr * dw
    b = b - lr * db

    if epoch % 20 == 0:
        print(f"Epoch{epoch:3d} | Loss:{loss:.4f} | w:{w[0,0]:.4f} | b:{b:.4f}")

print(f"\nLearned: w ={w[0,0]:.4f}, b ={b:.4f}")
print(f"True:    w = 3.0000, b = 4.0000")
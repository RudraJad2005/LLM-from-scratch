import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)                # Beautiful closed-form!

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# ============================================================
#  XOR DATASET — 4 examples, 2 features each
# ============================================================
X = np.array([[0, 0],                   # 4 training examples (rows)
              [0, 1],
              [1, 0],
              [1, 1]])                   # shape: (4, 2)

y = np.array([[0],                       # XOR labels
              [1],
              [1],
              [0]])                      # shape: (4, 1)

# ============================================================
#  INITIALIZE WEIGHTS (random small values)
# ============================================================
np.random.seed(42)                       # For reproducibility

# Layer 1: input (2) → hidden (4)
W1 = np.random.randn(2, 4) * 0.5        # (2, 4) — 2 inputs, 4 hidden neurons
b1 = np.zeros((1, 4))                    # (1, 4) — one bias per hidden neuron

# Layer 2: hidden (4) → output (1)
W2 = np.random.randn(4, 1) * 0.5        # (4, 1) — 4 hidden neurons, 1 output
b2 = np.zeros((1, 1))                   # (1, 1) — one bias for output

# ============================================================
#  TRAINING LOOP
# ============================================================
learning_rate = 0.5
epochs = 10000
losses = []                              # Track loss for plotting

for epoch in range(epochs):
    # ------- FORWARD PASS -------
    # Layer 1
    z1 = X @ W1 + b1                     # (4, 2) @ (2, 4) + (1, 4) = (4, 4)
    a1 = relu(z1)                        # Apply ReLU activation

    # Layer 2
    z2 = a1 @ W2 + b2                    # (4, 4) @ (4, 1) + (1, 1) = (4, 1)
    a2 = sigmoid(z2)                     # Final prediction ŷ, shape (4, 1)

    # ------- COMPUTE LOSS (Binary Cross-Entropy) -------
    epsilon = 1e-8                        # Prevent log(0)
    loss = -np.mean(y * np.log(a2 + epsilon) + (1 - y) * np.log(1 - a2 + epsilon))
    losses.append(loss)

    # ------- BACKWARD PASS -------
    m = X.shape[0]                        # Number of examples (4)

    # Output layer gradients
    dz2 = a2 - y                          # (4, 1) — gradient of loss w.r.t. z2
                                          # (this is the simplified sigmoid+BCE gradient!)
    dW2 = (a1.T @ dz2) / m               # (4, 1) — average over examples
    db2 = np.sum(dz2, axis=0, keepdims=True) / m   # (1, 1)

    # Hidden layer gradients (chain rule in action!)
    da1 = dz2 @ W2.T                     # (4, 4) — propagate error back
    dz1 = da1 * relu_derivative(z1)      # (4, 4) — gradient through ReLU
    dW1 = (X.T @ dz1) / m                # (2, 4) — weight gradients
    db1 = np.sum(dz1, axis=0, keepdims=True) / m   # (1, 4)

    # ------- UPDATE WEIGHTS (Gradient Descent) -------
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # Print progress every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch{epoch:5d} | Loss:{loss:.6f}")


print("\n=== Final Predictions ===")
print(f"Input [0,0] →{a2[0,0]:.4f}  (expected 0)")
print(f"Input [0,1] →{a2[1,0]:.4f}  (expected 1)")
print(f"Input [1,0] →{a2[2,0]:.4f}  (expected 1)")
print(f"Input [1,1] →{a2[3,0]:.4f}  (expected 0)")

rounded = np.round(a2)
accuracy = np.mean(rounded == y) * 100 
print(f"\nAccuracy:{accuracy:.0f}%")


plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title("Training Loss Over Time — XOR Problem")

plt.xlabel("Epoch")

plt.ylabel("Binary Cross-Entropy Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("xor_loss_curve.png", dpi=100)
plt.show()


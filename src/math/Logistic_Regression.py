import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
    )

y = y.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def sigmoid(z):

    z = np.clip(z, -500, 500)  # Clip values to avoid overflow
    return 1 / (1 + np.exp(-z))
    

n_features = X_train.shape[1]
w = np.zeros((n_features, 1))
b = 0.0
lr = 0.1
epochs = 200
n = X_train.shape[0]
losses = []

for epoch in range(epochs):
    
    # Forward pass
    z = X_train @ w + b
    y_pred = sigmoid(z)

    # Binary cross-entropy loss

    eps = 1e-8
    loss = -np.mean(
        y_train * np.log(y_pred + eps) + 
        (1 - y_train) * np.log(1 - y_pred + eps)
        )

    losses.append(loss)

    # Compute gradients

    error = y_pred - y_train
    dw = (1 / n) * (X_train.T @ error)
    db = (1 / n) * np.sum(error)

    # Updating parameters 
    w = w  - lr * dw
    b = b - lr * db

    if epoch % 40 == 0:
        acc = np.mean((y_pred >= 0.5).astype(int) == y_train)
        print(f"Epoch{epoch:3d} | Loss:{loss:.4f} | Accuracy:{acc:.2%}")

y_test_pred = sigmoid(X_test @ w + b)
test_acc = np.mean((y_test_pred >= 0.5).astype(int) == y_test)
print(f"\nTest Accuracy:{test_acc:.2%}")
print(f"Learned weights: w ={w.flatten()}, b ={b:.4f}")



clf = sklearn_lr = LogisticRegression(random_state=42)
clf.fit(X_train, y_train.ravel())

print(f"\nOur implementation:    w ={w.flatten()}, b ={b:.4f}")
print(f"sklearn's result:      w ={clf.coef_.flatten()}, b ={clf.intercept_[0]:.4f}")
print(f"\nOur test accuracy:{test_acc:.2%}")
print(f"sklearn test accuracy:{clf.score(X_test, y_test.ravel()):.2%}")


# Plotting decision boundary

def plot_decision_boundary(X, y, w, b):
    """Visualize the learned decision boundary"""
    plt.figure(figsize=(8, 6))

    # Plot data points
    plt.scatter(X[y.ravel()==0, 0], X[y.ravel()==0, 1],
                c='blue', marker='o', label='Class 0', alpha=0.6)
    plt.scatter(X[y.ravel()==1, 0], X[y.ravel()==1, 1],
                c='red', marker='x', label='Class 1', alpha=0.6)

    # Plot decision boundary: w1*x1 + w2*x2 + b = 0
    # Solve for x2: x2 = -(w1*x1 + b) / w2
    x1_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    x2_boundary = -(w[0, 0] * x1_range + b) / w[1, 0]

    plt.plot(x1_range, x2_boundary, 'g-', linewidth=2, label='Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

plot_decision_boundary(X_train, y_train, w, b)
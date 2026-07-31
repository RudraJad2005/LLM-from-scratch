# Forward pass from scratch using numpy

import numpy as np 


x = np.array([[1.0], [0.5]])       

# Layer 1
W1 = np.array([[0.3, -0.1],
               [0.5,  0.4]])        
b1 = np.array([[0.1], [-0.2]]) 


z1 = W1.T @ x + b1
h = np.maximum(0, z1)

# Layer 2

W2 = np.array([[0.6, 0.7]])
b2 = np.array([[-0.3]])

z2 = W2 @ h + b2
y_hat = 1 / (1+np.exp(-z2))

print(f"Output of the neural network (y_hat): {y_hat[0, 0]:.4f}")



# Mean Squared Error Loss

def mse_loss(y_true, y_pred):
    return np.mean((y_true + y_pred) ** 2)


# Binary Cross-Entropy Loss

def binary_cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# Softmax + Cross-Entropy Loss for multi-class classification

def softmax_cross_entropy_loss(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum()

raw_scores = np.array([2.0, 1.0, 0.5])
probs = softmax_cross_entropy_loss(raw_scores)
print(probs)


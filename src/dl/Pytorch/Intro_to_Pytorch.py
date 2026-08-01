import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

# Creating tensors just like numpy arrays

a = torch.tensor([1, 2, 3])
b = torch.zeros(3, 4)  # 3 rows, 4 columns of zeros (3x4 matrix)
c = torch.randn(2, 3) # 2 rows, 3 columns of random numbers from normal distribution
d = torch.from_numpy(np.array([1, 2]))

# Operations on tensors

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
print(x + y)  # Element-wise addition
print(x @ y)  # Dot product (matrix multiplication)
print(x.reshape(3,1)) # Reshape tensor to 3 rows, 1 column

if torch.cuda.is_available():
    x_gpu = x.to('cuda')
    print(x_gpu.device)  # Tensor on GPU


# Autograd and gradients

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

y = x**2 + 2*x + 1

y.backward(torch.ones_like(y))

print(x.grad)

# A more complex example

W = torch.tensor([[0.5, -0.3],
                  [0.2,  0.8]], requires_grad=True)
x = torch.tensor([[1.0], [2.0]])                      # Input (no grad needed)
b = torch.tensor([[0.1], [-0.1]], requires_grad=True)

z = W @ x + b
a = torch.relu(z)

loss = a.sum()

loss.backward()
print("dL/dW:\n", W.grad)         # How loss changes w.r.t. each weight
print("dL/db:\n", b.grad)         # How loss changes w.r.t. each bias


# rebuilding the XOR NuralNet - but this time using Pytorch's nn.Module:

class XORNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(2, 4) # 2 input -> 4 hidden neurons
        self.output = nn.Linear(4, 1) # 4 hidden -> 1 output


    def forward(self, n):

        n = torch.relu(self.hidden(n)) # Hidden layer + ReLU
        n = self.sigmoid(self.output(n))  # Output layer + Sigmoid
        return n


# DATASET

X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# SETUP

model = XORNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

# TRAINING LOOP

for epoch in range(1000):

    prediction = model(x)

    loss = criterion(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 2000 == 0:
        print(f"Epoch{epoch:5d} | Loss:{loss.item():.6f}")


# TESTING WITH NO GRAD

with torch.no_grad():

    preds = model(x)
    print("\nFinal Predictions:")
    for i in range(4):
        print(f"{X[i].tolist()} →{preds[i].item():.4f} (expected{y[i].item():.0f})")


# Creating a model

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(), # Activation
    nn.Linear(4, 1),
    nn.Sigmoid() # Output activation
)

# Loss functions

nn.MSELoss() # Mean squared error (Linear regression)
nn.BCELoss() # Binary cross-entropy (Binary classification)
nn.CrossEntropyLoss() # Softmax + Cross-Entropy (multi-class)

# Optimizers

optim.SGD(model.parameters(), lr=0.01)
optim.Adam(model.parameters(), lr=0.001)

# Useful operations

model.parameters()      # Iterator over all learnable weights
model.eval()         # Switch to evaluation mode (disables dropout, etc.)
model.train()         # Switch back to training mode
torch.no_grad()    # Context manager — disables gradient computation

import numpy as np 

np.random.seed(42)
X = 2 * np.random.rand(100, 1)           # 100 samples, 1 feature, range [0, 2]
y = 4 + 3 * X + np.random.randn(100, 1)



w = np.random.randn(1, 1) 
b = 0.0                    
lr = 0.1                    # learning rate (alpha)
epochs = 100                # number of passes through all data
n = len(X)                  # number of samples
losses = []  


for epoch in range(epochs):
 
    y_pred = X @ w + b                   

    loss = np.mean((y - y_pred) ** 2)  
    losses.append(loss)

    error = y - y_pred                    
    dw = (-2 / n) * (X.T @ error)     
    db = (-2 / n) * np.sum(error)       


    w = w - lr * dw                      
    b = b - lr * db                   

    if epoch % 20 == 0:
        print(f"Epoch{epoch:3d} | Loss:{loss:.4f} | w:{w[0,0]:.4f} | b:{b:.4f}")






import numpy as np 

def normal_equation(X, y):

    return np.linalg.inv(X.T @ X) @ X.T @ y

normal_equation(np.array([[1, 1], [1, 2], [2, 2], [2, 3]]), np.array([6, 8, 9, 11]))

print("Normal Equation Result:", normal_equation)
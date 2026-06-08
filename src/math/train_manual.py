from src.math.autodiff import Value
from src.math.nn import MLP

# 1. Setup the model and data
model = MLP(3, [4, 4, 1])
inputs = [Value(2.0), Value(3.0), Value(-1.0)]
desired_output = 1.0

# 2. The Training Loop
for k in range(20):
    
    # --- STEP A: FORWARD PASS ---
    # Goal: Get the model's current guess for the 'inputs'
    # yp = ...
    
    # --- STEP B: CALCULATE LOSS ---
    # Goal: Calculate how far 'yp' is from 'desired_output'
    # We usually use (guess - target)^2
    # loss = ...
    
    # --- STEP C: BACKWARD PASS ---
    # Goal: Reset gradients to 0.0 first, then calculate new gradients
    # 1. Zero out grads: for p in model.parameters(): p.grad = 0.0
    # 2. Trigger backprop: loss.backward()
    
    # --- STEP D: UPDATE (The Learning Step) ---
    # Goal: Change every parameter in 'model.parameters()' slightly
    # rule: p.data += -0.01 * p.grad
    
    # print(f"Iteration {k}, loss: {loss.data}, prediction: {yp.data}")
    pass
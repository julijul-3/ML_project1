import numpy as np
import implementations

y =  np.array([[0.1], [0.3], [0.5]])
tx = np.array([[2.3, 3.2], [1.0, 0.1], [1.4, 2.3]])
GAMMA = 0.1

expected_w = np.array([[0.413044], [0.875757]])
w, loss = implementations.mean_squared_error_gd(y, tx, expected_w, 0, GAMMA)

print("w" + w)
print("loss" + loss)
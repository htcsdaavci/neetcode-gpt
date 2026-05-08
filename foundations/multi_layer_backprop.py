import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        result = {}
        #for i in range(len(W1)):
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)

        z1 = np.dot(W1, x) + b1
        a1 = np.maximum(0, z1) # relu mask
        z2 = np.dot(W2, a1) + b2
        a2 = np.maximum(0, z2)

        loss = np.mean(np.square(z2 - y_true))

        # backward pass
        if y_true.ndim > 0:
            n = len(y_true)
        else:
            n = 1
        
        d_z2 = 2 * (z2 - y_true) / n
        d_W2 = np.dot(d_z2.reshape(-1,1), a1.reshape(1,-1))
        db2 = d_z2

        d_a1 = np.dot(d_z2.reshape(1,-1), W2)
        d_a1 = d_a1.flatten()
        d_z1 = d_a1 * (z1 > 0).astype(float)
        d_W1 = np.dot(d_z1.reshape(-1,1), x.reshape(1,-1))
        db1 = d_z1

        return {
            "loss": np.round(float(loss), 4),
            "dW1": np.round(d_W1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(d_W2, 4).tolist(),
            "db2": np.round(db2, 4).tolist(),
        }
        pass

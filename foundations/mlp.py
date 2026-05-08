import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        # x = np.array(x)
        #weights = np.array(weights)
        #biases = np.array(biases)
        def relu_derivative(x):
            return 0 if x < 0 else 1
        def relu(x):
            return np.maximum(0,x)
        hidden_layer = x
        output = [len(hidden_layer)]
        for i in range(len(weights) - 1) :
            hidden_layer = relu(np.dot(hidden_layer, weights[i]) + biases[i])
        
        output = np.dot(hidden_layer, weights[-1]) + biases[-1]
        return np.round(output,5)    
from typing import List
import random
import math

class DenseLayerLite:
    def __init__(self, input_dim: int, output_dim: int, activation: str = 'none'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation.lower()

        self.weights = [
            [random.gauss(0, 0.1) for _ in range(input_dim)]
            for _ in range(output_dim)
        ]
        self.biases = [0.0 for _ in range(output_dim)]
        
        self.input = []
        self.output = []

    def initialize(self, input_shape):
        self.output_shape = self.output_dim 

    def activate(self, x: List[float]) -> List[float]:
        if self.activation == "relu":
            return [val if val > 0 else 0 for val in x]
        elif self.activation == "softmax":
            max_val = max(x)
            exps = [math.exp(val - max_val) for val in x]
            total = sum(exps)
            return [e / total for e in exps]
        return x

    def activation_derivative(self, activated: List[float]) -> List[float]:
        if self.activation == "relu":
            return [1.0 if val > 0 else 0.0 for val in activated]
        elif self.activation == "softmax":
            return [val * (1 - val) for val in activated]
        return [1.0 for _ in activated]

    def forward(self, x: List[float]) -> List[float]:
        self.input = x
        raw_output = [
            sum(x[j] * self.weights[i][j] for j in range(self.input_dim)) + self.biases[i]
            for i in range(self.output_dim)
        ]
        self.output = self.activate(raw_output)
        return self.output

    def backward(self, grad_output: List[float], lr: float) -> List[float]:
        d_out = self.activation_derivative(self.output)
        grad_input = [0.0 for _ in range(self.input_dim)]

        for i in range(self.output_dim):
            delta = grad_output[i] * d_out[i]
            for j in range(self.input_dim):
                grad_input[j] += delta * self.weights[i][j]
                self.weights[i][j] -= lr * delta * self.input[j]
            self.biases[i] -= lr * delta

        return grad_input

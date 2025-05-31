import math
from typing import List, Union


NestedList = Union[float, List['NestedList']]


class ActivationLayer:
    def __init__(self, mode: str):
        self.mode = mode.lower()
        self.inputs: NestedList = None

    def apply(self, func, x: NestedList) -> NestedList:
        if isinstance(x, list):
            return [self.apply(func, xi) for xi in x]
        return func(x)

    def forward(self, inputs: NestedList) -> NestedList:
        self.inputs = inputs
        if self.mode == 'relu':
            return self.apply(lambda x: max(0.0, x), inputs)
        elif self.mode == 'sigmoid':
            return self.apply(lambda x: 1 / (1 + math.exp(-max(min(x, 100), -100))), inputs)
        elif self.mode == 'softmax':
            return [self.softmax(vec) for vec in inputs]
        else:
            raise ValueError(f"Unsupported activation: {self.mode}")

    def backward(self, grad_output: NestedList, learning_rate: float = 0.0) -> NestedList:
        if self.mode == 'relu':
            return self.apply_pair(lambda g, x: g if x > 0 else 0, grad_output, self.inputs)
        elif self.mode == 'sigmoid':
            sigmoid_out = self.forward(self.inputs)
            return self.apply_pair(lambda g, y: g * y * (1 - y), grad_output, sigmoid_out)
        elif self.mode == 'softmax':
            soft_out = self.forward(self.inputs)
            grads = []
            for out, grad in zip(soft_out, grad_output):
                dot = sum(o * g for o, g in zip(out, grad))
                grads.append([o * (g - dot) for o, g in zip(out, grad)])
            return grads
        else:
            raise ValueError(f"Unsupported activation: {self.mode}")

    def softmax(self, x: List[float]) -> List[float]:
        max_val = max(x)
        exp_vals = [math.exp(i - max_val) for i in x]
        total = sum(exp_vals)
        return [v / total for v in exp_vals]

    def apply_pair(self, func, a: NestedList, b: NestedList) -> NestedList:
        if isinstance(a, list) and isinstance(b, list):
            return [self.apply_pair(func, ai, bi) for ai, bi in zip(a, b)]
        return func(a, b)

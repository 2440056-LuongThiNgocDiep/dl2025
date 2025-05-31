from typing import List

class Layer:
    def forward(self, input_data: List) -> List:
        raise NotImplementedError
    
    def backward(self, grad_output: List, lr: float) -> List:
        raise NotImplementedError
from typing import List

class Flatten3DTo1D:
    def __init__(self):
        self.shape = (0, 0, 0)

    def initialize(self, input_shape):
        self.input_shape = input_shape
        c, h, w = input_shape
        self.output_shape = c * h * w

    def forward(self, input_3d: List[List[List[float]]]) -> List[float]:
        c, h, w = len(input_3d), len(input_3d[0]), len(input_3d[0][0])
        self.shape = (c, h, w)

        return [input_3d[ch][i][j] for ch in range(c) for i in range(h) for j in range(w)]

    def backward(self, grad_output: List[float], lrlr: float = 0.0) -> List[List[List[float]]]:
        c, h, w = self.shape
        reshaped = [[[0.0 for _ in range(w)] for _ in range(h)] for _ in range(c)]

        for idx in range(len(grad_output)):
            ch = idx // (h * w)
            rem = idx % (h * w)
            i = rem // w
            j = rem % w
            reshaped[ch][i][j] = grad_output[idx]

        return reshaped

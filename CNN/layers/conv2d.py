import random
import math
from typing import List, Tuple, Union

class Conv2D:
    def __init__(self, input_channels: int, filters: int, kernel_height: Union[int, Tuple[int, int]] = 3, stride: int = 1,
                 padding: str = 'valid', activation: str = 'relu'):
        self.input_channels = input_channels
        self.filters = filters
        
        if isinstance(kernel_height, int):
            self.kernel_height = kernel_height
            self.kernel_width = kernel_height
        else:
            self.kernel_height = kernel_height[0]
            self.kernel_width = kernel_height[1]
        
        self.stride = stride
        self.padding = padding.lower()
        self.activation = activation.lower()
        
        self.weights = self.init_kernels()
        self.bias = [0.0] * self.filters
        
        self.input_shape = None
        self.output_shape = None
        self.inputs = None
    
    def init_kernels(self) -> List[List[List[List[float]]]]:
        return [[
            [[random.uniform(-0.1, 0.1) for _ in range(self.kernel_height)]
             for _ in range(self.kernel_height)]
            for _ in range(self.input_channels)]
            for _ in range(self.filters)]
    
    def pad_input(self, input_data: List[List[List[float]]], pad: int) -> List[List[List[float]]]:
        c, h, w = len(input_data), len(input_data[0]), len(input_data[0][0])
        padded = []
        for ch in range(c):
            channel = []
            for i in range(h + 2*pad):
                row = []
                for j in range(w + 2*pad):
                    if pad <= i < h+pad and pad <= j < w+pad:
                        row.append(input_data[ch][i - pad][j - pad])
                    else:
                        row.append(0.0)
                channel.append(row)
            padded.append(channel)
        return padded
    
    def initialize(self, input_shape: Tuple[int, int, int]):
        self.input_shape = input_shape
        c, h, w = input_shape
        
        if self.padding == 'same':
            out_h = math.ceil(h / self.stride)
            out_w = math.ceil(w / self.stride)
        else:
            out_h = math.floor((h - self.kernel_height) / self.stride) + 1
            out_w = math.floor((w - self.kernel_height) / self.stride) + 1
        
        self.output_shape = (self.filters, out_h, out_w)
    
    def forward(self, input_data: List[List[List[float]]]) -> List[List[List[float]]]:
        self.inputs = input_data

        c, h, w = self.input_shape
        f, out_h, out_w = self.output_shape

        pad_h = self.kernel_height // 2 if self.padding == 'same' else 0
        pad_w = self.kernel_width // 2 if self.padding == 'same' else 0

        x_padded = self.pad_input(input_data, max(pad_h, pad_w))

        output = [[[0.0 for _ in range(out_w)] for _ in range(out_h)] for _ in range(f)]

        for filter_idx in range(f):
            for i_out in range(out_h):
                for j_out in range(out_w):
                    i_in = i_out * self.stride
                    j_in = j_out * self.stride
                    val = self.bias[filter_idx]
                    for ch in range(c):
                        for ki in range(self.kernel_height):
                            for kj in range(self.kernel_width):
                                val += (x_padded[ch][i_in + ki][j_in + kj] *
                                        self.weights[filter_idx][ch][ki][kj])
                    if self.activation == 'relu':
                        val = max(0, val)
                    output[filter_idx][i_out][j_out] = val

        return output

    
    def backward(self, grad_output: List[List[List[float]]], learning_rate: float) -> List[List[List[float]]]:
        c, h, w = self.input_shape
        f, out_h, out_w = self.output_shape

        pad_h = self.kernel_height // 2 if self.padding == 'same' else 0
        pad_w = self.kernel_width // 2 if self.padding == 'same' else 0

        grad_input_padded = [[[0.0 for _ in range(w + 2*max(pad_h, pad_w))] for _ in range(h + 2*max(pad_h, pad_w))] for _ in range(c)]

        for filter_idx in range(f):
            for i_out in range(out_h):
                for j_out in range(out_w):
                    grad_val = grad_output[filter_idx][i_out][j_out]
                    i_in = i_out * self.stride
                    j_in = j_out * self.stride
                    for ch in range(c):
                        for ki in range(self.kernel_height):
                            for kj in range(self.kernel_width):
                                grad_input_padded[ch][i_in + ki][j_in + kj] += (
                                    grad_val * self.weights[filter_idx][ch][ki][kj]
                                )

        if max(pad_h, pad_w) > 0:
            pad = max(pad_h, pad_w)
            grad_input = [
                [row[pad:-pad] for row in channel[pad:-pad]]
                for channel in grad_input_padded
            ]
        else:
            grad_input = grad_input_padded

        return grad_input


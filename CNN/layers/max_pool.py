class MaxPool:
    def __init__(self, pool_size=(2, 2), stride=None):
        self.pool_size = pool_size
        self.stride = stride or pool_size[0]
        self.cache = {}
        self.output_shape = None

    def initialize(self, input_shape):
        c, h, w = input_shape
        ph, pw = self.pool_size
        sh = sw = self.stride
        out_h = (h - ph) // sh + 1
        out_w = (w - pw) // sw + 1
        self.output_shape = (c, out_h, out_w)

    def forward(self, inputs):
        c, h, w = len(inputs), len(inputs[0]), len(inputs[0][0])
        ph, pw = self.pool_size
        sh = sw = self.stride

        out_h = (h - ph) // sh + 1
        out_w = (w - pw) // sw + 1

        output = []
        max_indices = []

        for ch in range(c):
            out_channel = []
            idx_channel = []
            for i in range(out_h):
                row = []
                idx_row = []
                for j in range(out_w):
                    max_val = float('-inf')
                    max_pos = (0, 0)
                    for pi in range(ph):
                        for pj in range(pw):
                            y = i * sh + pi
                            x = j * sw + pj
                            if y < h and x < w:
                                val = inputs[ch][y][x]
                                if val > max_val:
                                    max_val = val
                                    max_pos = (y, x)
                    row.append(max_val)
                    idx_row.append(max_pos)
                out_channel.append(row)
                idx_channel.append(idx_row)
            output.append(out_channel)
            max_indices.append(idx_channel)

        self.cache = {
            "input_shape": (c, h, w),
            "max_indices": max_indices
        }
        return output

    def backward(self, grad_output, learning_rate=0.0):
        c, h, w = self.cache["input_shape"]
        out_h = len(grad_output[0])
        out_w = len(grad_output[0][0])
        grad_input = [[[0.0 for _ in range(w)] for _ in range(h)] for _ in range(c)]
        max_indices = self.cache["max_indices"]

        for ch in range(c):
            for i in range(out_h):
                for j in range(out_w):
                    y, x = max_indices[ch][i][j]
                    grad_input[ch][y][x] = grad_output[ch][i][j]

        return grad_input

from typing import List, Tuple
import math
from PIL import Image
import matplotlib.pyplot as plt
from layers import conv2d, max_pool, flatten, dense
import os


class CNN:
    def __init__(self, config_path: str):
        self.layers = []
        self.input_shape = None
        self.loss_history = []
        self.load_config(config_path)

    def load_config(self, path: str):
        with open(path, 'r') as file:
            lines = [line.strip() for line in file.readlines() if line.strip()]

        num_layers = int(lines[0])
        dense_seen = 0
        current_dim = None

        for idx, line in enumerate(lines[1:num_layers+1]):
            tokens = line.split()
            layer_type = tokens[0]
            args = dict(item.split('=') for item in tokens[1:])
            
            if layer_type == 'Input':
                if idx != 0:
                    raise ValueError("Input layer must be the first layer")
                shape = tuple(map(int, args['shape'].split('x')))
                if len(shape) != 3:
                    raise ValueError("Input shape must be in CxHxW format")
                self.input_shape = shape
                current_dim = shape
                continue

            if layer_type == 'Conv2D':
                c, h, w = current_dim
                filters = int(args['filters'])
                kernel = tuple(map(int, args['kernel'].split('x')))
                stride = int(args.get('stride', 1))
                padding = args.get('padding', 'valid')
                activation = args.get('activation', 'relu')

                kh, kw = kernel

                if padding == 'same':
                    out_h = math.ceil(h / stride)
                    out_w = math.ceil(w / stride)
                else:
                    out_h = math.floor((h - kh) / stride) + 1
                    out_w = math.floor((w - kw) / stride) + 1

                layer = conv2d.Conv2D(
                    input_channels=c,
                    filters=filters,
                    kernel_height=kernel,
                    stride=stride,
                    padding=padding,
                    activation=activation
                )
                self.layers.append(layer)
                current_dim = (filters, out_h, out_w)

            elif layer_type == 'MaxPool2D':
                c, h, w = current_dim
                pool_size = tuple(map(int, args['pool'].split('x')))
                stride = int(args.get('stride', pool_size[0]))
                ph, pw = pool_size

                out_h = math.floor((h - ph) / stride) + 1
                out_w = math.floor((w - pw) / stride) + 1

                layer = max_pool.MaxPool(pool_size, stride)
                self.layers.append(layer)
                current_dim = (c, out_h, out_w)

            elif layer_type == 'Dense':
                dense_seen += 1
                units = int(args['units'])
                activation = args.get('activation', 'softmax' if idx == num_layers - 1 else 'relu')
                if dense_seen == 1:
                    self.layers.append(flatten.Flatten3DTo1D())
                    input_dim = current_dim[0] * current_dim[1] * current_dim[2]
                    current_dim = input_dim

                self.layers.append(dense.DenseLayerLite(input_dim, units, activation))
                current_dim = units

    def load_from_folder_structure(self, base_folder: str) -> Tuple[List[List[List[float]]], List[int]]:
        images = []
        labels = []

        for label_name in sorted(os.listdir(base_folder)):
            label_path = os.path.join(base_folder, label_name)
            if not os.path.isdir(label_path):
                continue
            try:
                label = int(label_name)
            except ValueError:
                continue

            count = 0
            for file_name in os.listdir(label_path):
                if count >= 20:
                    break
                if file_name.lower().endswith('.png'):
                    img_path = os.path.join(label_path, file_name)
                    img_data = self.load_image(img_path)
                    images.append(img_data)
                    labels.append(label)
                    count += 1

        return images, labels


    def initialize(self):
        shape = self.input_shape
        for layer in self.layers:
            layer.initialize(shape)
            shape = layer.output_shape

    def forward_pass(self, data: List[List[List[float]]]) -> List[float]:
        for layer in self.layers:
            data = layer.forward(data)
        return data

    def cross_entropy(self, predicted: List[float], label: int) -> Tuple[float, List[float]]:
        target = [1.0 if i == label else 0.0 for i in range(len(predicted))]
        loss = -sum(t * math.log(max(p, 1e-10)) for t, p in zip(target, predicted))
        grad = [p - t for p, t in zip(predicted, target)]
        return loss, grad

    def train(self, images: List[List[List[float]]], labels: List[int], epochs: int, lr: float):
        self.initialize()
        for epoch in range(epochs):
            total_loss = 0.0
            for img, label in zip(images, labels):
                pred = self.forward_pass(img)
                loss, grad = self.cross_entropy(pred, label)
                total_loss += loss

                for layer in reversed(self.layers):
                    grad = layer.backward(grad, lr)

            avg_loss = total_loss / len(images)
            self.loss_history.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    def load_image(self, path: str) -> List[List[List[float]]]:
        img = Image.open(path).convert('L')
        return [[[float(img.getpixel((x, y))) / 255.0 for y in range(img.height)] for x in range(img.width)]]

    def load_data(self, folder: str, label_path: str) -> Tuple[List[List[List[float]]], List[int]]:
        images, labels = [], []
        with open(label_path, 'r') as f:
            for line in f:
                fname, lbl = line.strip().split()
                img_data = self.load_image(f"{folder}/{fname}")
                images.append(img_data)
                labels.append(int(lbl))
        return images, labels

    def predict(self, image: List[List[List[float]]]) -> int:
        output = self.forward_pass(image)
        return output.index(max(output))


    def evaluate(self, test_images: List[List[List[float]]], test_labels: List[int]) -> float:
        correct = 0
        total = len(test_labels)

        for img, label in zip(test_images, test_labels):
            pred = self.predict(img)
            if pred == label:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0
        return accuracy*100

    def plot_loss(self):
        if not self.loss_history:
            print("No training history available to plot.")
            return
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history)
        plt.title("Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig("plots/loss_plot.jpg")
        plt.show()

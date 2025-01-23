# Neural Network Implementation

A simple feedforward neural network implementation in Python. This project supports customizable layer structures and activation functions with backpropagation for training.

## Features
- Fully connected layers with customizable structures.
- Supports the following activation functions:
  - `sigmoid`
  - `relu`
  - `tanh`
  - `softmax`
- Lightweight dependencies: only `numpy` is required.

---

## Requirements

- **File:** `nn.py` (contains the `NeuralNetwork` implementation)
- **Library:** `numpy`  
  Install it using:  
  ```bash
  pip install numpy
  
## How to Use

Initialize the Network
- To create a neural network with 4 input nodes, a hidden layer with 40 nodes using ReLU activation, and an output layer with 2 nodes using softmax activation:
'''
from nn import NeuralNetwork
import numpy as np

# Initialize the neural network
network = NeuralNetwork([4, 40, 2], ["relu", "softmax"])
# Prepare Input and Labels
# Define the input data (x) and corresponding labels (x_labels):
x = np.array([
    [0, 1, 0, 1],
    [0, 1, 1, 1]
])

x_labels = np.array([
    [0, 1],
    [1, 0]
])
# Train the Network
for row in range(len(x)):
    mse, prediction = network.learn(x[row], x_labels[row])
    print(f"Mean Squared Error: {mse}")
    print(f"Prediction: {prediction}")
# Inspect Weights and Biases
for layer in network.layers:
    weights = layer.weights
    biases = layer.bias
    print(f"Weights:\n{weights}")
    print(f"Biases:\n{biases}")

'''

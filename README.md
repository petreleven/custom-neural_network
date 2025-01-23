Ah, understood! If you want to display images that are already in your GitHub repository, you can link to them using Markdown syntax. Here's how you can modify your README to include the images:

---

# Neural Network Implementation

A simple feedforward neural network implementation in Python. This project supports customizable layer structures and activation functions with backpropagation for training.

## Features

- **Fully connected layers** with customizable structures.
- **Supports the following activation functions:**
  - `sigmoid`
  - `relu`
  - `tanh`
  - `softmax`
- **Lightweight dependencies**: only `numpy` is required.

---

## Requirements

- **File:** `nn.py` (contains the `NeuralNetwork` implementation)
- **Library:** `numpy`  
  Install it using:  
  ```bash
  pip install numpy
  ```

---

## How to Use

### 1. Initialize the Network

To create a neural network with:
- 4 input nodes
- A hidden layer with 40 nodes using ReLU activation
- An output layer with 2 nodes using softmax activation

```python
from nn import NeuralNetwork
import numpy as np

# Initialize the neural network
network = NeuralNetwork([4, 40, 2], ["relu", "softmax"])
```

### 2. Prepare Input Data and Labels

Define the input data `x` and corresponding labels `x_labels`:

```python
x = np.array([
    [0, 1, 0, 1],
    [0, 1, 1, 1]
])

x_labels = np.array([
    [0, 1],
    [1, 0]
])
```

### 3. Train the Network

Loop through the input data and train the network using the `learn` method. It returns the mean squared error and the prediction for each input.

```python
for row in range(len(x)):
    mse, prediction = network.learn(x[row], x_labels[row])
    print(f"Mean Squared Error: {mse}")
    print(f"Prediction: {prediction}")
```

### 4. Inspect Weights and Biases

Inspect the weights and biases for each layer of the neural network:

```python
for layer in network.layers:
    weights = layer.weights
    biases = layer.bias
    print(f"Weights:\n{weights}")
    print(f"Biases:\n{biases}")
```

---

## Images of Network Learning

Here are some images showing the neural network learning:

![Training Progress 1](https://github.com/petreleven/custom-neural_network/blob/main/669.png)
*Training step 1: Initial weights and biases.*

![Training Progress 2](https://github.com/petreleven/custom-neural_network/blob/main/learning.gif)


---


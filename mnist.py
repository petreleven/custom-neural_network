import sys
from keras.datasets import mnist
from visualizer.neuralnetwork import NeuralNetwork
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
images = x_train[:200].reshape(200, 28 * 28) / 255

labels = y_train[0:200]
labels_probability = np.zeros((len(labels), 10))
for i, v in enumerate(labels):
    labels_probability[i][v] = 1

neuralnet: NeuralNetwork = NeuralNetwork(
    nodes_per_layer=[28 * 28, 20, 10], activations=["sigmoid", "softmax"]
)

for i in range(100):
    error = 0.0
    for j in range(images.shape[0]):
        error += neuralnet.learn(inputs=images[j], targets=labels_probability[j])

    error /= images.shape[0]
    sys.stdout.write(f"Epoch {i}\n")
    sys.stdout.write(f"Error is {error}\n")

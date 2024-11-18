from typing import List, TypeVar, Union
from typing_extensions import NoDefault, Tuple
from numpy.typing import NDArray
import numpy as np
import time

class Layer:
    def __init__(self, nodes_in: int, nodes_out: int, type_of_activation: str):
        self.weights: NDArray = np.random.normal(
            -0.5, 0.5, (nodes_out, nodes_in)
        ).astype(np.float32)
        self.nodes_in = nodes_in
        self.nodes_out = nodes_out
        self.outputs = np.zeros(self.nodes_out)
        self.inputs = np.zeros(self.nodes_in)
        self.msq_error = np.zeros(self.nodes_out)
        self.bias = np.random.normal(-1.0, 1.0, nodes_out)
        self.type_of_activation = type_of_activation
        self.lr = 0.1

    def _compute_prediction(self, inputs: NDArray) -> NDArray:
        # print(f"inputs shape :{len(inputs)} weightshapes {self.weights.shape}")

        """output: NDArray = np.zeros(self.nodes_out)
        for i in range(self.nodes_out):
            for j in range(self.nodes_in):
                output[i] += inputs[j] * self.weights[i][j]
            output[i] += self.bias[i]
        """
        output = np.dot(self.weights, inputs) + self.bias
        return self.activation(output)

    def activation(self, activation_input: NDArray) -> NDArray:
        # sigmoid
        if self.type_of_activation == "sigmoid":
            return 1 / (1 + np.exp(-activation_input))
        # tanh
        elif self.type_of_activation == "tanh":
            return np.tanh(activation_input)  # tanh
        # softmax
        elif self.type_of_activation == "softmax":
            temp = np.exp(activation_input)
            output = temp / np.sum(temp)
            return output

    def activation_derivative(
        self,
        output: Union[NDArray, float],
    ):
        # sigmoid
        if self.type_of_activation == "sigmoid":
            return output * (1 - output)
        # tanh
        elif self.type_of_activation == "tanh":
            return 1 - (output**2)
        # softmax
        elif self.type_of_activation == "softmax":
            return output * (1 - output)

    def _predict(self, inputs: NDArray):
        assert len(inputs) == self.nodes_in
        self.outputs: NDArray = self._compute_prediction(inputs=inputs)

    def get_prediction(self, inputs: NDArray) -> NDArray:
        assert len(inputs) == self.nodes_in
        self.inputs = inputs
        self.outputs = self._compute_prediction(inputs=inputs)
        return self.outputs

    def get_msq_error(self, targets: NDArray):
        assert len(targets) == len(self.outputs)
        for i in range(self.nodes_out):
            self.msq_error[i] = (self.outputs[i] - targets[i]) ** 2

        return np.mean(self.msq_error)

    def _get_deltas(self, targets: NDArray):
        deltas = np.zeros(self.nodes_out)
        for i in range(self.nodes_out):
            error = self.outputs[i] - targets[i]
            deltas[i] = error * self.activation_derivative(
                self.outputs[i]
            )
        return deltas

    def _get_weight_deltas(self, targets: NDArray, inputs: NDArray):
        assert len(targets) == len(self.outputs)
        deltas = self._get_deltas(targets)
        weight_deltas = np.zeros(shape=self.weights.shape)
        for i in range(self.nodes_out):
            for j in range(self.nodes_in):
                weight_deltas[i][j] = deltas[i] * inputs[j]
        return weight_deltas

    def learn(self, targets: NDArray, inputs: NDArray):
        self._predict(inputs)
        weight_deltas = self._get_weight_deltas(targets, inputs)
        # UPDATE WEIGHTS
        if not np.all(np.isfinite(weight_deltas)):
            print("Warning: Invalid values in weight_deltas, skipping update.")
            return
        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                self.weights[i][j] -= weight_deltas[i][j] * self.lr
        # UPDATE BIASES
        bias_deltas = self._get_deltas(targets)
        self.bias -= self.lr * np.mean(bias_deltas)
        return self.get_msq_error(targets=targets)

    def learn_as_nn_layer(self, deltas: NDArray[float]):
        weights_delta = np.zeros((self.nodes_out, self.nodes_in))
        for i in range(self.nodes_in):
            for j in range(self.nodes_out):
                weights_delta[j][i] = self.inputs[i] * (
                    deltas[j] * self.activation_derivative(output=self.outputs[j])
                )
        self.weights -= self.lr * weights_delta
        self.bias -= self.lr * deltas


NDint_float = TypeVar("NDint_float", NDArray[int], NDArray[float])


class NeuralNetwork:
    # 2 3 2
    def __init__(self, nodes_per_layer: List[int], activations: List[str]):
        self.layers: List[Layer] = []
        for i in range(len(nodes_per_layer) - 1):
            new_layer = Layer(
                nodes_in=nodes_per_layer[i],
                nodes_out=nodes_per_layer[i + 1],
                type_of_activation=activations[i],
            )
            self.layers.append(new_layer)

    def forward_propagate(self, inputs: NDint_float) -> NDint_float:
        propageted_inputs: NDint_float = inputs
        for layer in self.layers:
            output = layer.get_prediction(propageted_inputs)
            propageted_inputs = output
        return propageted_inputs  # this is now the final output

    def get_error(self, prediction: NDint_float, targets: NDint_float) -> NDint_float:
        errors: NDint_float = np.zeros(len(targets))
        for i in range(len(targets)):
            errors[i] = prediction[i] - targets[i]
        return errors

    def get_delta(self, prediction: NDint_float, targets: NDint_float) -> NDint_float:
        errors = self.get_error(prediction=prediction, targets=targets)
        deltas = np.array(len(errors))

        return deltas

    def learn(self, inputs: NDint_float, targets: NDint_float) -> float:
        assert self.layers[0].nodes_in == len(inputs)
        assert self.layers[len(self.layers) - 1].nodes_out == len(targets)
        prediction: NDint_float = self.forward_propagate(inputs)
        # compute deltas
        endDeltas: List[List[float]] = []
        errors = self.get_error(prediction=prediction, targets=targets)
        endDeltas.append(errors)

        for i in range(len(self.layers) - 2, -1, -1):
            current_deltas: NDArray[float] = np.array(endDeltas[-1])
            next_layer = self.layers[i + 1]
            weights_T = next_layer.weights.T
            hidden_delta = np.dot(weights_T, current_deltas)
            endDeltas.append(hidden_delta)

        if False:  # only for debugging
            print("Targets are " + str(targets))
            print("prediction " + str(prediction))
            print("biases are " + str(self.layers[-1].bias))
            print("weights are " + str(self.layers[-1].weights))
            print("deltas are " + str(endDeltas))

        endDeltas = endDeltas[::-1]
        # backpropagate
        for i, layer in enumerate(self.layers):
            self.layers[i].learn_as_nn_layer(deltas=endDeltas[i])

        return np.mean(self.get_error(prediction=prediction, targets=targets) ** 2)

from abc import abstractmethod
import sys
from typing import Dict
from numpy.random.mtrand import randint
from typing_extensions import List, Tuple
import pygame
import random
import numpy as np
import pprint
from drawmanager import DrawManager
import random
from abc import abstractmethod, ABC

from neuralnetwork import Layer, NeuralNetwork
import json

error_file = open("data.txt", "w")
file_epoch = []
file_error = []


class Graph(ABC):
    resolution = (560, 560)
    fps = 60
    screen = pygame.display.set_mode(resolution)
    screen.fill((0, 0, 0))
    clock = pygame.time.Clock()
    drawmanager = DrawManager(resoultion=resolution)
    BRUSH_SIZE = int(1 / 28 * resolution[0])
    RERENDER: bool = False

    def __init__(self):
        pygame.init()
        self.safe: List[Dict[str, float]] = self.create_training(0, 290, 0, 500)
        self.poisonus: List[Dict[str, float]] = self.create_training(300, 500, 0, 300)
        self.lastdraw = 0

    def draw_data(self):
        pygame.surfarray.blit_array(self.screen, self.drawmanager.get_pixel_buffer())
        steps = self.resolution[0] / 10
        for y in range(10):
            for x in range(10):
                pygame.draw.line(
                    self.screen,
                    (255, 255, 255, 100),
                    (steps * x, 0),
                    (steps * x, steps * y),
                )
                pygame.draw.line(
                    self.screen,
                    (255, 255, 255, 100),
                    (0, steps * y),
                    (self.resolution[1] - 1, steps * y),
                )
        for center in self.safe:
            pygame.draw.circle(
                self.screen,
                (0, 0, 255, 100),
                center=(center["x"], center["y"]),
                radius=10,
            )
        for center in self.poisonus:
            pygame.draw.circle(
                self.screen,
                (255, 0, 0, 100),
                center=(center["x"], center["y"]),
                radius=10,
            )

    def update(self):
        self.draw_data()

    def update_nn(self, layer: NeuralNetwork):
        self._set_pixels_buffer_from_nn(layer)

    def create_training(self, startx: int, endx: int, starty: int, endy: int):
        result: List[Dict[str, float]] = []
        for i in range(40):
            x = np.random.randint(startx, endx)
            y = np.random.randint(starty, endy)
            result.append({"x": float(x), "y": float(y)})
        return result

    def _set_pixels_buffer(self):
        pixel_positions: List[Tuple[int, int]] = [pygame.mouse.get_pos()]
        for y in range(-self.BRUSH_SIZE, self.BRUSH_SIZE):
            for x in range(-self.BRUSH_SIZE, self.BRUSH_SIZE):
                next_pixel = (pixel_positions[0][0] + x, pixel_positions[0][1] + y)
                if (
                    next_pixel[0] >= 0 and next_pixel[0] <= self.resolution[0] - 1
                ) and (next_pixel[1] >= 0 and next_pixel[1] <= self.resolution[1] - 1):
                    pixel_positions.append(next_pixel)

    def _set_pixels_buffer_from_nn(self, layer: NeuralNetwork):
        res = 10
        pixel_update_size = self.resolution[0] // res
        for y in range(pixel_update_size):
            for x in range(pixel_update_size):
                network_input = np.array(
                    [x * res / self.resolution[0], y * res / self.resolution[1]]
                )
                output = layer.forward_propagate(network_input).ravel()
                pixel_color = []
                intensity = int(min(200, abs(max(output)) * 200))

                if output[1] > output[0]:
                    pixel_color = [0, 0, intensity]
                else:
                    pixel_color = [intensity, 0, 0]

                self.drawmanager.pixel_buffer[
                    y * res : (y + 1) * res,
                    x * res : (x + 1) * res,
                ] = pixel_color
                """if x*res==100  and y*res==400:
                    print(f"output {output}")
                    print(f"pixel {pixel_color}")"""


custom_graph = Graph()
singlelayer = NeuralNetwork([2, 4, 6, 2], ["sigmoid", "tanh", "softmax"])

lastTick = 0
lastTick2 = 0
epoch = 0
new_weights = []
new_bias = []
while 1:
    # get events
    # custom_graph.handle_input([slider1, slider2])
    """singlelayer.weights[0][0] = slider1.value
    singlelayer.weights[0][1] = slider2.value"""
    # clear & update screen
    custom_graph.screen.fill((0, 0, 0))
    if pygame.time.get_ticks() - lastTick > 30:
        e = 0
        for p in custom_graph.safe:
            inputs = np.array(
                [
                    p["x"] / custom_graph.resolution[0],
                    p["y"] / custom_graph.resolution[1],
                ]
            )
            msq = singlelayer.learn(
                inputs=inputs,
                targets=np.array([1, 0]),
            )
            if msq:
                e += msq
        for p in custom_graph.poisonus:
            inputs = np.array(
                [
                    p["x"] / custom_graph.resolution[0],
                    p["y"] / custom_graph.resolution[1],
                ]
            )
            msq = singlelayer.learn(inputs=inputs, targets=np.array([0, 1]))
            if msq:
                e += msq
        e = e / (len(custom_graph.safe) + len(custom_graph.poisonus))
        print(f"Epoch :{epoch}")
        print(f"Error :{e}")
        file_epoch.append(epoch)
        file_error.append(e)
        error_file.seek(0)
        error_file.truncate()
        json.dump({"epoch": file_epoch, "error": file_error}, error_file)
        error_file.flush()
        epoch += 1
        lastTick = pygame.time.get_ticks()

    if pygame.time.get_ticks() - lastTick2 > 60:
        custom_graph.update_nn(singlelayer)
        lastTick2 = pygame.time.get_ticks()

    if epoch == 5000:
        break

    custom_graph.draw_data()
    # slider1.draw(custom_graph.screen)
    # slider2.draw(custom_graph.screen)
    # update widgets & screen
    pygame.display.flip()
    custom_graph.clock.tick(30)


print("*" * 20)
data_in = [
    custom_graph.safe[0]["x"] / custom_graph.resolution[0],
    custom_graph.safe[0]["y"] / custom_graph.resolution[1],
]
print(f"Datain {data_in}")
pred = singlelayer.forward_propagate(np.array(data_in))
print(f"Prediction: {pred}")

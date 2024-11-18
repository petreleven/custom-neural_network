from typing_extensions import List
import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class DrawManager:
    def __init__(self, resoultion: Tuple[int, int]):
        self._resolution = resoultion
        self.pixel_buffer: NDArray[np.uint8] = np.random.randint(
            0, 1, (self._resolution[0], self._resolution[1], 3), dtype=np.uint8
        )

    def get_pixel_buffer(self):
        return self.pixel_buffer

    def update_pixel_buffer_c(
        self, pixel_positions: List[Tuple[int, int]], color: List[int]
    ):
        for pixel in pixel_positions:
            self.pixel_buffer[pixel[0], pixel[1]] = np.array(color)

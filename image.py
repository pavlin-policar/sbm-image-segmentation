from typing import Tuple

import numpy as np


def pixel2node(i: int, j: int, image: np.ndarray) -> int:
    """Convert pixel indices i, j to a single integer node id."""
    height = image.shape[0]
    return height * i + j


def node2pixel(node: int, image: np.ndarray) -> Tuple[int, int]:
    """Convert a single integer node id to pixel indices i, j."""
    height = image.shape[0]
    return node // height, node % height

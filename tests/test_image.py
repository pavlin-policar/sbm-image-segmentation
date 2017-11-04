import unittest

import numpy as np

from data_provider import BSDS
from image import pixel2node, node2pixel


class TestImage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = BSDS.load('3096')

    def test_pixel2node_node2pixel(self):
        # Pick a random image index i, j
        i, j = 318, 233
        image_height, image_width = self.image.shape[:2]
        assert i < image_width and j < image_height, \
            'Selected indices out of image bounds'

        # Convert the image indices to node id
        node_id = pixel2node(i, j, self.image)
        self.assertTrue(
            node_id < np.prod(self.image.shape[:2]),
            'Node id is larger than number of nodes'
        )

        # Convert the node id back to the image coordinates
        i_, j_ = node2pixel(node_id, self.image)

        self.assertEqual(
            i,
            i_,
            'Transformed `i` coordinate does not match initial value',
        )
        self.assertEqual(
            j,
            j_,
            'Transformed `j` coordinate does not match initial value',
        )

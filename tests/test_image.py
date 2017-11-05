import unittest

import numpy as np

from data_provider import BSDS
from image import pixel2node, node2pixel, pixel_vec, l2_distance, \
    pixel_similarity


class TestImagePixelConversion(unittest.TestCase):
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


class TestImageSimilarity(unittest.TestCase):
    def test_pixel_vec(self):
        image = np.array([
            [[0, 0, 0], [0, 0, 0]],
            [[100, 50, 0], [100, 75, 0]],
            [[255, 0, 0], [0, 255, 0]],
        ])

        dist_same = l2_distance(pixel_vec(0, 0, image), pixel_vec(0, 1, image))
        self.assertEqual(dist_same, 0, 'Distance of identical pixels in not 0')

        dist_similar = l2_distance(pixel_vec(1, 0, image), pixel_vec(1, 1, image))
        dist_different = l2_distance(pixel_vec(2, 0, image), pixel_vec(2, 1, image))

        self.assertTrue(
            dist_same < dist_similar,
            'dist_same is not smaller than dist_similar'
        )
        self.assertTrue(
            dist_similar < dist_different,
            'dist_similar is not smaller than dist different'
        )

    def test_pixel_distance_term(self):
        # All pixels are the same
        image = np.zeros((5, 5, 3))
        similarities = np.zeros((5, 5))

        # Take the central pixel, and compute the distances from there
        x, y = 2, 2

        for idx in np.ndindex(similarities.shape):
            similarities[idx] = pixel_similarity((y, x), idx, image, sigma_x=8, sigma_i=1)

        # Manual debugging
        # np.set_printoptions(precision=3, suppress=True)
        # print(similarities)

        # Check symmetry
        # 1 pixel away
        self.assertTrue(similarities[1, 2] == similarities[3, 2])
        self.assertTrue(similarities[2, 1] == similarities[2, 3])
        # 2 pixels away
        self.assertTrue(similarities[0, 2] == similarities[4, 2])
        self.assertTrue(similarities[2, 0] == similarities[2, 4])

    def test_pixel_similarity(self):
        # All pixels are the same
        image = np.zeros((5, 5, 3))
        image[2:, :] = [1, 1, 1]
        similarities = np.zeros((5, 5))

        # Take the central pixel, and compute the distances from there
        x, y = 2, 2

        for idx in np.ndindex(similarities.shape):
            similarities[idx] = pixel_similarity((y, x), idx, image, sigma_x=8, sigma_i=1)
        # Due to indexing going x, y instead of y, x, transpose similarities
        similarities = similarities.T

        # Manual debugging
        # np.set_printoptions(precision=3, suppress=True)
        # print(similarities)

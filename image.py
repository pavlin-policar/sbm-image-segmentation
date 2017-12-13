from colorsys import rgb_to_hsv
from typing import Tuple, Iterable

import fire
import graph_tool as gt
import numpy as np
from graph_tool import inference


def pixel2node(i: int, j: int, image: np.ndarray) -> int:
    """Convert pixel indices i, j to a single integer node id."""
    height = image.shape[0]
    return height * j + i


def node2pixel(node: int, image: np.ndarray) -> Tuple[int, int]:
    """Convert a single integer node id to pixel indices i, j."""
    height = image.shape[0]
    return node % height, node // height


def l2_distance(vec1: Iterable, vec2: Iterable) -> float:
    """Compute the l2 distance between two vectors."""
    vec1, vec2 = np.array(vec1), np.array(vec2)
    diff = vec1 - vec2
    return np.sqrt(diff.dot(diff))


def pixel_vec(y: int, x: int, image: np.ndarray) -> np.ndarray:
    """Get a feature vector represnting a pixel on the image.

    If a grayscale image is given, return the pixel intensity value, otherwise
    return the HSV transform for colored images.

    References
    ----------
    ..  [1] Browet, Arnaud, Pierre-Antoine Absil, and Paul Van Dooren.
        "Community Detection for Hierarchical Image Segmentation." IWCIA. Vol.
        11. 2011.

    """
    if image.ndim == 2:
        return image[y, x]
    return rgb_to_hsv(*image[y, x])


def pixel_similarity(pixel1, pixel2, image, sigma_x, sigma_i):
    """Compute the similarity between two pixels.

    Parameters
    ----------
    pixel1 : Tuple[int, int]
    pixel2 : Tuple[int, int]
    image : np.ndarray
    sigma_x : float
        User parameter that determines the strength of distance. A lower value
        induces faster decay i.e. distances will drop off exponentially,
        whereas a larger values will induce a slower decay.
    sigma_i : float
        User parameter that determines the strength of pixel similarity. A
        lower value induces faster decay i.e. the higher the difference in
        pixel color, the lower the similarity whereas a larger value will
        induce a slower decay.

    Returns
    -------
    float

    """
    feature_vec_1 = pixel_vec(*pixel1, image)
    feature_vec_2 = pixel_vec(*pixel2, image)
    return (
        # Distance term
        np.exp(-l2_distance(pixel1, pixel2) ** 2 / sigma_x ** 2) *
        # Pixel similarity term
        np.exp(-l2_distance(feature_vec_1, feature_vec_2) ** 2 / sigma_i ** 2)
    )


def image_to_graph(image, d_max, sigma_x, sigma_i):
    """Convert an image to a weighted graph.

    Parameters
    ----------
    image : np.ndarray
    d_max : float
        The maximum distance between two pixels that will get an edge.
    sigma_x : float
        User defined parameter for inter-pixel distance weight.
    sigma_i : float
        User defined parameter for inter-pixel similarity weight.

    Return
    ------
    gt.Graph
        A weighted undirected graph with a `weight` edge property.

    References
    ----------
    ..  [1] Shi, Jianbo, and Jitendra Malik. "Normalized cuts and image
        segmentation." IEEE Transactions on pattern analysis and machine
        intelligence 22.8 (2000): 888-905.
    ..  [2] Browet, Arnaud, Pierre-Antoine Absil, and Paul Van Dooren.
        "Community Detection for Hierarchical Image Segmentation." IWCIA. Vol.
        11. 2011.

    """
    height, width = image.shape[:2]
    edges = {}

    for y, x in np.ndindex((height, width)):
        d_max_int = int(np.ceil(d_max))
        for j in range(y - d_max_int, y + d_max_int):
            # If we're outside the boundaries of the image skip iteration
            if j < 0 or j >= height:
                continue
            for i in range(x - d_max_int, x + d_max_int):
                # If we're outside the boundaries of the image skip iteration
                if i < 0 or i >= width:
                    continue

                # We won't connect the pixel with itself
                if j == y and i == x:
                    continue

                # Check if euclidean distance exceeds `d_max` then skip
                distance = l2_distance((j, i), (y, x))
                if distance > d_max:
                    continue

                # Add the edge between the pixels
                node1, node2 = pixel2node(y, x, image), pixel2node(j, i, image)
                edges[frozenset((node1, node2))] = pixel_similarity(
                    (y, x), (j, i), image, sigma_x=sigma_x, sigma_i=sigma_i
                )

    # Create the corresponding graph object
    graph = gt.Graph(directed=False)
    graph.ep.weight = graph.new_edge_property('double')
    # Add the appropriate number of vertices to the graph
    graph.add_vertex(height * width)
    graph.add_edge_list(edges)

    for edge in graph.edges():
        graph.ep.weight[edge] = edges[frozenset((edge.source(), edge.target()))]

    return graph


def sbm_segmentation(graph: gt.Graph, image: np.ndarray) -> np.ndarray:
    """Use the stochastic block model to obtain a segmentation of the image."""
    state = inference.minimize_blockmodel_dl(
        graph,
        state_args=dict(recs=[graph.ep.weight], rec_types=['real-exponential']),
    )
    blocks = state.get_blocks()

    segmentation = np.zeros(image.shape[:2], dtype=int)
    for node in graph.vertices():
        segmentation[node2pixel(int(node), image)] = blocks[node]

    return segmentation


def hsbm_segmentation(graph: gt.Graph, image: np.ndarray) -> np.ndarray:
    """Use the hierarchical stochastic block model to obtain multiple
    segmentations of the image at different levels."""
    state = inference.minimize_nested_blockmodel_dl(
        graph,
        state_args=dict(recs=[graph.ep.weight], rec_types=['real-exponential']),
    )
    segmentations = []

    # Get the first level segmentation
    iterator = iter(state.get_levels())
    bottom_level = next(iterator)
    blocks = bottom_level.get_blocks()
    segmentation = np.zeros(image.shape[:2], dtype=int)
    for node in graph.vertices():
        segmentation[node2pixel(int(node), image)] = blocks[node]
    segmentations.append(segmentation)

    # For the further layers, we need to change segmentation mask ids properly
    for level in iterator:
        blocks = level.get_blocks()
        segmentation = np.array(segmentations[-1])
        for mask_id in np.unique(segmentation):
            new_mask_id = blocks[mask_id]
            segmentation[segmentation == mask_id] = new_mask_id
        segmentations.append(segmentation)

    return segmentations


if __name__ == '__main__':
    fire.Fire()

import fire
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.cm as cm

import graph_tool.draw
from data_provider import BSDS
from image import image_to_graph, node2pixel, sbm_segmentation, pixel2node

sns.set('paper', 'whitegrid')


def image_graph(image_id: str):
    image_id = str(image_id)
    top, left = 100, 200
    height, width = 100, 100
    image = BSDS.load(image_id)[top:top + height, left:left + width]
    graph = image_to_graph(image, 2, np.sqrt(2), 4)

    colors = graph.new_vertex_property('vector<double>')
    for node in graph.vertices():
        colors[node] = image[node2pixel(int(node), image)] / 255

    positions = graph.new_vertex_property('vector<float>')
    for node in graph.vertices():
        positions[node] = reversed(node2pixel(int(node), image))

    graph_tool.draw.graph_draw(
        graph, pos=positions, vertex_fill_color=colors, vertex_size=3,
        edge_pen_width=graph.ep.weight,
    )


def image_partition(image_id: str):
    image_id = str(image_id)
    top, left = 100, 200
    height, width = 100, 100
    image = BSDS.load(image_id)[top:top + height, left:left + width]
    graph = image_to_graph(image, 3, np.sqrt(2), 4)

    segmentation = sbm_segmentation(graph, image)
    colors = graph.new_vertex_property('vector<float>')
    for idx in np.ndindex(segmentation.shape[:2]):
        colors[pixel2node(*idx, image)] = cm.terrain(segmentation[idx] / np.max(segmentation))

    positions = graph.new_vertex_property('vector<float>')
    for node in graph.vertices():
        positions[node] = reversed(node2pixel(int(node), image))

    graph_tool.draw.graph_draw(graph, pos=positions, vertex_fill_color=colors, vertex_size=5)


def true_segmentation(image_id: str):
    image_id = str(image_id)
    image, segmentation = BSDS.load(image_id), BSDS.segmentation(image_id)

    ax = plt.subplot(121)
    ax.set_title('Original image')
    ax.imshow(image)
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax = plt.subplot(122)
    ax.set_title('True segmentation mask')
    ax.imshow(segmentation)
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.show()


if __name__ == '__main__':
    fire.Fire()

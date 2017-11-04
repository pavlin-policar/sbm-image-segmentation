import fire
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import graph_tool.draw
from data_provider import BSDS
from image import image_to_graph, node2pixel

sns.set('paper', 'whitegrid')


def image_graph(image_id: str):
    image_id = str(image_id)
    image = BSDS.load(image_id)[:20, :20]
    graph = image_to_graph(image, 2, np.sqrt(2), 4)
    print(graph.vertex(43).out_degree())

    pos = graph.new_vertex_property('vector<float>')
    for node in graph.vertices():
        pos[node] = node2pixel(int(node), image)
    graph_tool.draw.graph_draw(graph, pos=pos)


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

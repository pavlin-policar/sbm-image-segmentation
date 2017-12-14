import fire
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.cm as cm
from skimage import segmentation

import graph_tool.draw
from data_provider import BSDS
from image import image_to_graph, node2pixel, sbm_segmentation, pixel2node, \
    hsbm_segmentation

sns.set('paper', 'whitegrid')


def image_graph(image_id: str):
    image_id = str(image_id)
    top, left = 0, 0
    height, width = 200, 150
    image = BSDS.load(image_id)[top:top + height, left:left + width]
    # image = BSDS.load(image_id)
    graph = image_to_graph(image, 3, 5, 4)

    # colors = graph.new_vertex_property('vector<double>')
    # for node in graph.vertices():
    #     colors[node] = image[node2pixel(int(node), image)] / 255

    positions = graph.new_vertex_property('vector<float>')
    for node in graph.vertices():
        positions[node] = reversed(node2pixel(int(node), image))

    # graph_tool.draw.graph_draw(
    #     graph, pos=positions, vertex_fill_color=colors, vertex_size=3,
    #     edge_pen_width=graph.ep.weight,
    # )

    graph_tool.draw.graph_draw(
        graph, pos=positions, vertex_size=1, edge_pen_width=graph.ep.weight,
    )


def sbm_partition(image_id: str, interactive: bool=False):
    image_id = str(image_id)
    # top, left = 200, 300
    # height, width = 80, 100
    # image = BSDS.load(image_id)[top:top + height, left:left + width]
    image = BSDS.load(image_id)
    graph = image_to_graph(image, 2, sigma_x=10, sigma_i=20)

    seg_mask = sbm_segmentation(graph, image)

    if interactive:
        colors = graph.new_vertex_property('vector<float>')
        for idx in np.ndindex(seg_mask.shape[:2]):
            colors[pixel2node(*idx, image)] = cm.terrain(seg_mask[idx] / np.max(seg_mask))

        positions = graph.new_vertex_property('vector<float>')
        for node in graph.vertices():
            positions[node] = reversed(node2pixel(int(node), image))

        graph_tool.draw.graph_draw(graph, pos=positions, vertex_fill_color=colors, vertex_size=5)
    else:
        ax = plt.subplot(121)
        ax.set_title('Original image')
        ax.imshow(image)
        ax.grid(False), ax.set_xticklabels([]), ax.set_yticklabels([])

        ax = plt.subplot(122)
        ax.set_title('SBM Segmentation')
        ax.imshow(segmentation.mark_boundaries(image, seg_mask, color=[1, 1, 1]))
        ax.grid(False), ax.set_xticklabels([]), ax.set_yticklabels([])

        plt.tight_layout()
        plt.show()


def hsbm_partition(image_id: str):
    image_id = str(image_id)
    # top, left = 200, 50
    # height, width = 20, 20
    # image = BSDS.load(image_id)[top:top + height, left:left + width]
    image = BSDS.load(image_id)
    graph = image_to_graph(image, 2, sigma_x=10, sigma_i=20)

    seg_masks = hsbm_segmentation(graph, image)

    ax = plt.subplot(421)
    ax.imshow(image)
    ax.grid(False), ax.set_xticklabels([]), ax.set_yticklabels([])

    for idx, seg_mask in enumerate(seg_masks[:7]):
        ax = plt.subplot(4, 2, idx + 2)
        ax.imshow(segmentation.mark_boundaries(image, seg_mask, color=[1, 1, 1]))
        ax.grid(False), ax.set_xticklabels([]), ax.set_yticklabels([])

    plt.tight_layout()
    plt.savefig('report/images/hsbm_%s_segmentation.png' % image_id, dpi=400)
    plt.show()


def true_segmentation(image_id: str):
    image_id = str(image_id)
    image, seg_mask = BSDS.load(image_id), BSDS.segmentation(image_id)[0]

    ax = plt.subplot(121)
    ax.set_title('Original image')
    ax.imshow(image)
    ax.grid(False), ax.set_xticklabels([]), ax.set_yticklabels([])

    ax = plt.subplot(122)
    ax.set_title('True segmentation mask')
    ax.imshow(seg_mask)
    ax.grid(False), ax.set_xticklabels([]), ax.set_yticklabels([])

    plt.show()


def slic_superpixel(image_id: str):
    image_id = str(image_id)
    image = BSDS.load(image_id)
    superpixels = segmentation.slic(image, n_segments=1000, compactness=10)

    plt.imshow(segmentation.mark_boundaries(image, superpixels, color=[1, 1, 1]))
    plt.grid(False), plt.xticks([]), plt.yticks([])
    plt.show()


def segmentation_differences(image_id: str):
    image_id = str(image_id)
    image, seg_masks = BSDS.load(image_id), BSDS.segmentation(image_id)

    ax = plt.subplot(331)
    ax.imshow(image)
    ax.grid(False), ax.set_xticklabels([]), ax.set_yticklabels([])

    for idx in range(2, 10):
        ax = plt.subplot(3, 3, idx)
        ax.imshow(seg_masks[idx - 2])
        ax.grid(False), ax.set_xticklabels([]), ax.set_yticklabels([])

    plt.tight_layout()
    plt.savefig('report/images/house_segmentations.png', dpi=400)
    plt.show()


if __name__ == '__main__':
    fire.Fire()

from typing import Optional

import fire
import graph_tool.draw
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
from scipy.ndimage import filters
from skimage import segmentation, transform

from data_provider import BSDS
from image import image_to_graph, node2pixel, sbm_segmentation, \
    hsbm_segmentation

# sns.set('paper', 'whitegrid')


def _read_image(image_id: str) -> np.ndarray:
    image_id = str(image_id)
    try:
        image = BSDS.load(image_id)

    # An assertion error is thrown if the image id is not found
    except AssertionError:
        image = plt.imread(image_id)

    return image


def image_graph(
        image_id: str, d_max: float=1.5, sigma_x: float=10, sigma_i: float=3,
        color: bool=False, blur: Optional[float]=None):

    image = _read_image(image_id)[:200, :200]

    if blur is not None:
        image = filters.gaussian_filter(image, sigma=blur)

    graph = image_to_graph(image, d_max=d_max, sigma_x=sigma_x, sigma_i=sigma_i)
    return

    positions = graph.new_vertex_property('vector<float>')
    for node in graph.vertices():
        positions[node] = reversed(node2pixel(int(node), image))

    fname = 'poster/koala_dmax_%d_sx_%d_si_%d' % (d_max, sigma_x, sigma_i)
    if blur is not None:
        fname += '_gauss_%d' % int(blur)
    if color:
        fname += '_color'
    fname += '.png'

    if color:
        colors = graph.new_vertex_property('vector<double>')
        for node in graph.vertices():
            colors[node] = image[node2pixel(int(node), image)] / 255

        graph_tool.draw.graph_draw(
            graph, pos=positions, vertex_fill_color=colors, vertex_size=1,
            edge_pen_width=graph.ep.weight,
            output=fname, output_size=image.shape,
        )
    else:
        graph_tool.draw.graph_draw(
            graph, pos=positions, vertex_size=0,
            edge_pen_width=graph.ep.weight,
            output=fname, output_size=image.shape,
        )


def sbm_partition(
        image_id: str, d_max: float=1.5, sigma_x: float=10, sigma_i: float=3,
        blur: Optional[float]=None, show_original: bool=False):

    image = _read_image(image_id)

    if blur is not None:
        image = filters.gaussian_filter(image, sigma=blur)

    graph = image_to_graph(image, d_max=d_max, sigma_x=sigma_x, sigma_i=sigma_i)

    seg_mask = sbm_segmentation(graph, image)

    fname = 'poster/koala_seg_dmax_%d_sx_%d_si_%d' % (d_max, sigma_x, sigma_i)
    if blur is not None:
        fname += '_gauss_%d' % int(blur)
    fname += '.png'

    if show_original:
        ax = plt.subplot(121)
        ax.set_title('Original image')
        ax.imshow(image)
        ax.grid(False), ax.set_xticklabels([]), ax.set_yticklabels([])

        ax = plt.subplot(122)
        ax.set_title('SBM Segmentation')
        ax.imshow(segmentation.mark_boundaries(image, seg_mask, color=[1, 1, 1]))
        ax.grid(False), ax.set_xticklabels([]), ax.set_yticklabels([])

        plt.tight_layout()
        plt.savefig(fname, dpi=1000)
    else:
        plt.imshow(segmentation.mark_boundaries(image, seg_mask, color=[1, 1, 1]))
        plt.savefig(fname, dpi=1000)


def hsbm_partition(
        image_id: str, d_max: float=1.5, sigma_x: float=10, sigma_i: float=3,
        blur: Optional[float]=None):
    image = _read_image(image_id)
    image = transform.rescale(image, 0.5)

    if blur is not None:
        image = filters.gaussian_filter(image, sigma=blur)

    graph = image_to_graph(image, d_max=d_max, sigma_x=sigma_x, sigma_i=sigma_i)

    seg_masks = hsbm_segmentation(graph, image)

    fname = 'poster/koala_hseg_dmax_%d_sx_%d_si_%d' % (d_max, sigma_x, sigma_i)
    if blur is not None:
        fname += '_gauss_%d' % int(blur)

    for idx, seg_mask in enumerate(seg_masks):
        plt.clf()
        plt.imshow(segmentation.mark_boundaries(image, seg_mask, color=[1, 1, 1]))
        plt.grid(False), plt.axis('off')
        plt.savefig('%s_%s.png' % (fname, idx))


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

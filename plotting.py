import fire
import matplotlib.pyplot as plt
import seaborn as sns
from data_provider import BSDS

sns.set('paper', 'whitegrid')


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
from collections import defaultdict
from os import listdir
from os.path import join
from typing import Dict

import numpy as np
from scipy.ndimage import imread

from data_provider import DATA_DIR

BASE_DIR = join(DATA_DIR, 'BSDS300')

SEGMENTATIONS = defaultdict(list)
for channel in listdir(join(BASE_DIR, 'human')):
    for user_dir in listdir(join(BASE_DIR, 'human', channel)):
        for image_seg in listdir(join(BASE_DIR, 'human', channel, user_dir)):
            SEGMENTATIONS[image_seg.split('.')[0]].append(
                join('human', channel, user_dir, image_seg))


def ids() -> Dict[str, str]:
    train_ = listdir(join(BASE_DIR, 'images', 'train'))
    files = {img_id.split('.')[0]: 'train/%s' % img_id for img_id in train_}
    test_ = listdir(join(BASE_DIR, 'images', 'test'))
    files.update({img_id.split('.')[0]: 'test/%s' % img_id for img_id in test_})
    return files


def load(image_id: str) -> np.ndarray:
    image_ids = ids()
    assert image_id in image_ids, 'Unrecognized image id'
    return imread(join(BASE_DIR, 'images', image_ids[image_id]))


def segmentation(image_id: str) -> np.ndarray:
    """Parse the image segmentation mask.

    Parameters
    ----------
    image_id : str

    Returns
    -------
    np.ndarray

    References
    ----------
    ..  [1] https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/seg-format.txt

    """
    assert image_id in SEGMENTATIONS, 'Unrecognized segmentation id'

    masks = []

    for segmentation_file in SEGMENTATIONS[image_id]:
        with open(join(BASE_DIR, segmentation_file), 'r') as f:
            # Ignore all the lines up to the `data` line
            for line in f:
                line = line.strip('\n')
                # Check for relevant fields in header
                if line.startswith('image'):
                    assert line.split(' ')[1] == image_id, \
                        'Segmentation image id does not match queried id'
                if line.startswith('width'):
                    width = int(line.split(' ')[1])
                elif line.startswith('height'):
                    height = int(line.split(' ')[1])
                elif line == 'data':
                    break

            seg_mask = np.zeros((height, width))

            # We've reached the data portion, so start parsing that
            for line in f:
                segment, row, col_start, col_end = map(int, line.strip('\n').split(' '))
                seg_mask[row, col_start:col_end + 1] = segment

            masks.append(seg_mask)

    return masks

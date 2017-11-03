from os import listdir
from os.path import join, realpath, dirname
from typing import Dict, List, Union

import fire
import numpy as np
from scipy.ndimage import imread

DATA_DIR = join(dirname(realpath(__file__)), 'data')
BSDS = join(DATA_DIR, 'BSDS300')
ADE20K = join(DATA_DIR, 'ADE20K_2016_07_26')
VOC2012 = join(DATA_DIR, 'VOCdevkit/VOC2012')


def bsds_ids() -> Dict[str, str]:
    train_ = listdir(join(BSDS, 'images', 'train'))
    files = {img_id.split('.')[0]: 'train/%s' % img_id for img_id in train_}
    test_ = listdir(join(BSDS, 'images', 'test'))
    files.update({img_id.split('.')[0]: 'test/%s' % img_id for img_id in test_})
    return files


def load_bsds(image_id: str=None) -> Union[np.ndarray, List[np.ndarray]]:
    image_ids = bsds_ids()
    # If a single image id is specified, load that one
    if image_id is not None:
        assert image_id in image_ids, 'Unrecognized image id'
        return imread(join(BSDS, 'images', image_ids[image_id]))

    # If no id is provided, load all the images
    return [imread(join(BSDS, 'images', image_ids[image_id])) for image_id in image_ids]


if __name__ == '__main__':
    fire.Fire()

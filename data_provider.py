from os import listdir
from os.path import join, realpath, dirname
from typing import Dict, List, Union

import fire
import numpy as np
from scipy.ndimage import imread

DATA_DIR = join(dirname(realpath(__file__)), 'data')
ADE20K = join(DATA_DIR, 'ADE20K_2016_07_26')
VOC2012 = join(DATA_DIR, 'VOCdevkit/VOC2012')


class DataProvider:
    BASE_DIR = ...

    @classmethod
    def ids(cls) -> Dict[str, str]:
        pass

    @classmethod
    def load(cls, image_id: str) -> np.ndarray:
        pass

    @classmethod
    def load_all(cls) -> List[np.ndarray]:
        return [cls.load(image_id) for image_id in cls.ids()]

    @classmethod
    def segmentation(cls):
        pass


class BSDS(DataProvider):
    BASE_DIR = join(DATA_DIR, 'BSDS300')

    _SEGMENTATIONS = {
        image_seg.split('.')[0]: join('human', channel, user_dir, image_seg)
        for channel in listdir(join(DATA_DIR, 'BSDS300', 'human'))
        for user_dir in listdir(join(DATA_DIR, 'BSDS300', 'human', channel))
        for image_seg in listdir(join(DATA_DIR, 'BSDS300', 'human', channel, user_dir))
    }

    @classmethod
    def ids(cls) -> Dict[str, str]:
        train_ = listdir(join(cls.BASE_DIR, 'images', 'train'))
        files = {img_id.split('.')[0]: 'train/%s' % img_id for img_id in train_}
        test_ = listdir(join(cls.BASE_DIR, 'images', 'test'))
        files.update({img_id.split('.')[0]: 'test/%s' % img_id for img_id in test_})
        return files

    @classmethod
    def load(cls, image_id: str) -> Union[np.ndarray, List[np.ndarray]]:
        image_ids = cls.ids()

        assert image_id in image_ids, 'Unrecognized image id'
        return imread(join(cls.BASE_DIR, 'images', image_ids[image_id]))

    @classmethod
    def segmentation(cls, image_id: str) -> np.ndarray:
        assert image_id in cls._SEGMENTATIONS, 'Unrecognized segmentation id'

        with open(join(cls.BASE_DIR, cls._SEGMENTATIONS[image_id]), 'r') as f:
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
                seg_mask[row, col_start:col_end] = segment

        return seg_mask


if __name__ == '__main__':
    fire.Fire()

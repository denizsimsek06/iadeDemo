import os
import sys

import numpy as np

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from mrcnn.eti import EtiDataset2
from mrcnn.get_image_size import get_image_size



ETI_DIR = os.path.join(ROOT_DIR, "datasets/packages")


data = EtiDataset2()

data.load_eti(ETI_DIR, 'train')

data.prepare()
print(get_image_size('/Users/denizsimsek/Desktop/iade/iadeDemo/datasets/packages/train/IMG_5649.JPG'))

mask, idasd = data.load_mask(69)
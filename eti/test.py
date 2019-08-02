import os
import numpy as np

from mrcnn.get_image_size import get_image_size

ROOT_DIR = os.path.abspath("../")

ETI_DIR = os.path.join(ROOT_DIR, "datasets/packages")


print(get_image_size('IMG_5608.jpg'))

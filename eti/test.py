import os
import numpy as np

import eti

ROOT_DIR = os.path.abspath("../")

ETI_DIR = os.path.join(ROOT_DIR, "datasets/packages")

dataset = eti.EtiDataset()
dataset.load_eti(ETI_DIR, "val")

# Must call before using the dataset
dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))



mask, class_id = dataset.load_mask(1)

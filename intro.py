import time
from imageio import imread
from keras_preprocessing.image import load_img, img_to_array, ImageDataGenerator, array_to_img
import matplotlib.pyplot as plt
import numpy as np

from loaddata import load_data

text_path = 'dataset/train.txt'
data_path = 'dataset/train/Packages'
preview_path = 'dataset/train/preview'
img_height = 500
img_width = 500

# x = load_data(data_path)
# x = x.transpose()

# training_path = x[:,0]

datagen_rescale = ImageDataGenerator(rescale=1. / 255.,
                                     rotation_range=45,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     brightness_range=(0.5, 1.5),
                                     shear_range=0.2,
                                     zoom_range=0.3,
                                     horizontal_flip=True,
                                     vertical_flip=True)

rescaled_data = datagen_rescale.flow_from_directory(data_path,
                                                    classes=['eti-tutku'],
                                                    target_size=(img_width,img_height),
                                                    shuffle=True,
                                                    seed=31)

i=0

for batch, label in rescaled_data:
    print(batch)
    i +=1
    if i>20:
        break

# = ImageDataGenerator(zca_whitening=True,
#                          rotation_range=45,
#                        height_shift_range=0.2,
#                        brightness_range=(0.5, 1.5),
#                      zoom_range=0.3,
#                     horizontal_flip=True,
#                    vertical_flip=True)

import numpy as np
from imageio import imread


def load_data(file_path):
    image_path = []
    product_label = []

    with open(file_path, 'r') as file:
        file_text = file.read().splitlines()
        for line in file_text:
            parts = line.split(', ')

            image_path.append('PackageImages/dataset/'+parts[0])
            product_label.append(parts[1])

    data = np.array([image_path, product_label])
    return data

def get_image(index, data):
    image = imread(data[[index]])

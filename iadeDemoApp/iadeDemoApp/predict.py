import sys
from os import path

import matplotlib

import skimage

PROJECT_DIR = path.abspath("/Users/denizsimsek/Desktop/iade/iadeDemo/")
sys.path.append(PROJECT_DIR)
from mrcnn import eti, utils, model as modellib, visualize

MODEL_DIR = path.abspath("/Users/denizsimsek/Desktop/iade/logs")
WEIGHTS_PATH = path.abspath("/Users/denizsimsek/Desktop/iade/logs/mask_rcnn_eti_train_0044.h5")
SAVE_DIR = path.abspath("/Users/denizsimsek/Desktop/iade/iadeDemo/iadeDemoApp/media/prediction/")
IMAGE_PATH=path.abspath("/Users/denizsimsek/Desktop/iade/iadeDemo/iadeDemoApp/media/upload/upload.jpg")

class InferenceConfig(eti.EtiConfigT(0, 0).__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.975
    IMG_MAX_DIM = 224
    IMG_MIN_DIM = 224


config = InferenceConfig(0, 0)

model_inference = modellib.MaskRCNN(mode="inference",
                                    model_dir=MODEL_DIR,
                                    config=config)

model_inference.load_weights(WEIGHTS_PATH, by_name=True)
model_inference.keras_model._make_predict_function()


def predict():
    matplotlib.use('Agg')
    image = skimage.io.imread(IMAGE_PATH)

    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    results = model_inference.detect([image], verbose=1)
    # get dictionary for first prediction
    r = results[0]
    visualize.save_image(image, 'prediction', r['rois'], r['masks'], r['class_ids'],
                         r['scores'], ['BG', 'tutku'], scores_thresh=config.DETECTION_MIN_CONFIDENCE,
                         save_dir=SAVE_DIR)
    return len(r['class_ids'])

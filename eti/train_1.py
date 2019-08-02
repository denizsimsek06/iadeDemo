import os
import sys

from imgaug import augmenters as aug

ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN# To find local version of the library
sys.path.append(ROOT_DIR)
from mrcnn.eti import EtiDataset, EtiConfigT
import mrcnn.model as modellib

# Path to trained weights file
ETI_DATA_PATH = os.path.join(ROOT_DIR, "datasets/packages")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs/mask_rcnn_coco.h5")
print(COCO_WEIGHTS_PATH)

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = EtiDataset()
    dataset_train.load_eti(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = EtiDataset()
    dataset_val.load_eti(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    
    augmentation = aug.Sequential([aug.Fliplr(0.5), aug.Flipud(0.5),
                                   aug.OneOf([aug.Affine(rotate=0),
                                              aug.Affine(rotate=90),
                                              aug.Affine(rotate=180),
                                              aug.Affine(rotate=270)]),
                                   aug.Sometimes(0.5, aug.Affine(rotate=(-10, 10))),
                                   aug.Add((-15, 15), per_channel=1)])

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads',
                augmentation=augmentation)
    print("Training layers 4+")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='4+',
                augmentation=augmentation)
    print("Training all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=50,
                layers='all',
                augmentation=augmentation)



if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect eti.')
    parser.add_argument('--dataset', required=False,
                        default=ETI_DATA_PATH,
                        metavar="/path/to/eti/dataset/",
                        help='Directory of the eti dataset')
    parser.add_argument('--weights', required=False,
                        default="coco",
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    config = EtiConfigT(n_imaget=46,n_imagev=11)

    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=args.logs)
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
    elif args.weights.lower() == "last":
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    if args.weights.lower() == "coco":
        model.load_weights(weights_path, by_name=True, exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    train(model)

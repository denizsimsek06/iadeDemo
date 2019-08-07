import os
import json
import numpy as np
import skimage.draw
from mrcnn.config import Config
from mrcnn.get_image_size import get_image_size
from mrcnn.utils import Dataset


############################################################
#  Configurations
############################################################

class EtiConfigT(Config):
    """Configuration for training on the eti  dataset.
    Derives from the base Config class and overrides some values.
    """
    NAME = "eti_train"

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    NUM_CLASSES = 1 + 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    BACKBONE = "resnet50"
    TRAIN_ROIS_PER_IMAGE = 64
    DETECTION_MAX_INSTANCES = 32
    MAX_GT_INSTANCES = 32

    def __init__(self, n_imaget, n_imagev):
        self.STEPS_PER_EPOCH = n_imaget * 3
        self.VALIDATION_STEPS = n_imagev * 3
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM,
                                         self.IMAGE_CHANNEL_COUNT])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM,
                                         self.IMAGE_CHANNEL_COUNT])

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES


class EtiConfig2(Config):
    """Configuration for training on the eti  dataset.
    Derives from the base Config class and overrides some values.
    """
    NAME = "eti_train2"

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    NUM_CLASSES = 1 + 2
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    BACKBONE = "resnet50"
    TRAIN_ROIS_PER_IMAGE = 128
    DETECTION_MAX_INSTANCES = 64
    MAX_GT_INSTANCES = 64

    def __init__(self, n_imaget, n_imagev):
        self.STEPS_PER_EPOCH = n_imaget * 3
        self.VALIDATION_STEPS = n_imagev * 3
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM,
                                         self.IMAGE_CHANNEL_COUNT])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM,
                                         self.IMAGE_CHANNEL_COUNT])

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES


############################################################
#  Dataset
############################################################

class EtiDataset(Dataset):

    def load_eti(self, dataset_dir, subset):
        """Load a subset of the eti dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        self.add_class("packages", 1, "tutku")

        assert subset in ["train", "val", "liltrain", "lilval"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = annotations["_via_img_metadata"]
        annotations = list(annotations.values())  # don't need the dict keys

        # Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.

            polygons = [r['shape_attributes'] for r in a['regions']]
            names = [l['region_attributes'] for l in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            width, height = get_image_size(image_path)
            self.add_image(
                "packages",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                names=names)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "packages":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        class_names = info["names"]

        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        class_ids = np.zeros([len(info["polygons"])])
        for i, p in enumerate(class_names):
            if p['name'] == 'tutku':
                class_ids[i] = 1

        # Return mask, and array of class IDs of each instance.
        return mask.astype(np.bool), np.asarray(class_ids.astype(np.int))

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "packages":
            return info["path"]
        else:
            super(EtiDataset, self).image_reference(self, image_id)

    def load_image(self, image_id):
        image = skimage.io.imread(self.image_info[image_id]['path'], plugin='pil')

        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image


class EtiDataset2(Dataset):

    def load_eti(self, dataset_dir, subset):
        """Load a subset of the eti dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        self.add_class("packages", 1, "tutku")
        self.add_class("packages", 2, "balik kraker")

        assert subset in ["train", "val", "liltrain", "lilval"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = annotations["_via_img_metadata"]
        annotations = list(annotations.values())  # don't need the dict keys

        # Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.

            polygons = [r['shape_attributes'] for r in a['regions']]
            names = [l['region_attributes'] for l in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            width, height = get_image_size(image_path)
            self.add_image(
                "packages",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                names=names)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "packages":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        class_names = info["names"]

        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        class_ids = np.zeros([len(info["polygons"])])
        for i, p in enumerate(class_names):
            if p['name'] == 'tutku':
                class_ids[i] = 1
            if p['name'] == 'balik kraker':
                class_ids[i] = 2

            #add more classes here if needed

        # Return mask, and array of class IDs of each instance.
        return mask.astype(np.bool), np.asarray(class_ids.astype(np.int))

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "packages":
            return info["path"]
        else:
            super(EtiDataset, self).image_reference(self, image_id)

    def load_image(self, image_id):
        image = skimage.io.imread(self.image_info[image_id]['path'], plugin='pil')

        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

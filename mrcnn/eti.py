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

class EtiConfig(Config):
    """Configuration for training on the eti  dataset.
    Derives from the base Config class and overrides some values.
    """
    NAME = "eti"

    NUM_CLASSES = 1 + 2

    IMAGE_RESIZE_MODE = "crop"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Smaller images for faster training
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_DIM = 512

    # Smaller anchors
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    # Few objects in image, use less ROIs per image

    # Number of classes (including background)

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 92
    VALIDATION_STEPS = 23

    USE_MINI_MASK = True

    # Skip detections with < 70% confidence
    DETECTION_MIN_CONFIDENCE = 0.7


############################################################
#  Dataset
############################################################

class EtiDataset(Dataset):

    def load_eti(self, dataset_dir, mode):
        """Load a subset of the eti dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        self.add_class("packages", 1, "tutku")
        self.add_class("packages", 2, "juice")

        dataset_dir = os.path.join(dataset_dir, mode)

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
            labels = [l['region_attributes'] for l in a['regions']]

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
                labels=labels)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        label_names = [n['name'] for n in info['labels']]
        class_ids = []
        for i, nom in enumerate(self.class_info):
            for name in label_names:
                if nom['name'] == name:
                    class_ids.append(i)

        # Return mask, and array of class IDs of each instance.
        return mask.astype(np.bool), np.asarray(class_ids)

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

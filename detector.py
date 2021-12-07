import tensorflow as tf
from threading import Semaphore
from PIL import Image
from skimage import measure
import numpy as np
import scripts


def check_box(box_area_limit=400, side_limit=25):
    def inner(box):
        min_x, min_y, max_x, max_y = box
        w = max_x - min_x
        h = max_y - min_y
        if h < side_limit:
            return False
        if w < side_limit:
            return False
        area = w * h
        if area < box_area_limit:
            return False
        return True
    return inner


def binarization_to_boxes(binarization, check_box=check_box()):
    regions, num_regions = measure.label(binarization, return_num=True)
    # find boxes around areas
    boxes = []
    for r in range(1, num_regions + 1):
        nonzero = np.nonzero(regions == r)
        min_x = min(nonzero[1])
        max_x = max(nonzero[1]) + 1
        min_y = min(nonzero[0])
        max_y = max(nonzero[0]) + 1
        # skip boxes that are too small or too elongated
        box = (min_x, min_y, max_x, max_y)
        if check_box(box):
            boxes.append(box)
    # if boxes have too much iou or overlap completely, join them
    boxes = scripts.boxes.join_boxes(boxes)
    return boxes


def get_boxes(probs, t=0.5, margin_ratio=1 / 16.):
    """Get boxes out of predictions."""
    # apply theshold
    assert probs.dtype == np.float32
    binarization = probs > t

    scripts.masks.clean_margins(binarization, margin_ratio)
    boxes = binarization_to_boxes(binarization)

    return boxes


def _signature_def_to_tensors(signature_def):
    g = tf.get_default_graph()
    return {k: g.get_tensor_by_name(v.name) for k, v in signature_def.inputs.items()}, \
           {k: g.get_tensor_by_name(v.name)
            for k, v in signature_def.outputs.items()}


def get_default_conf():
    return dict(num_parallel_predictions=1, threshold=256. / 2,
                margin_ratio=1. / 16., model_dir="model_weights")


def merge_conf_with_default(user_conf):
    config = get_default_conf()
    for key in user_conf.keys():
        if key not in config.keys():
            raise KeyError(
                f"Config key {key} not one of the config options. Default config is {config}")
    config.update(user_conf)
    return config


class Detector():
    def __init__(self, config={}):
        config = merge_conf_with_default(config)

        self.threshold = config["threshold"]
        self.margin_ratio = config["margin_ratio"]

        self.sess = tf.Session()

        loaded_model = tf.saved_model.loader.load(
            self.sess, ['serve'], config["model_dir"])

        assert 'serving_default' in list(loaded_model.signature_def)

        #input_dict_key = 'image'
        #signature_def_key = 'from_image:resized_output'

        input_dict_key = 'image'
        signature_def_key = 'from_image:resized_output'

        input_dict, output_dict = _signature_def_to_tensors(
            loaded_model.signature_def[signature_def_key])

        assert input_dict_key in input_dict.keys(), "{} not present in input_keys, " \
                                                    "possible values: {}".format(
                                                        input_dict_key, input_dict.keys())
        self._input_tensor = input_dict[input_dict_key]
        self._output_dict = output_dict
        self.sema = Semaphore(config["num_parallel_predictions"])

    def get_raw_prediction(self, image):
        image = Image.open(image).convert("RGB")
        width, height = image.size
        image = np.asarray(image)

        with self.sema:
            prediction_outputs = self.sess.run(self._output_dict, feed_dict={
                self._input_tensor: image})

        # Take only class '1' (class 0 is the background, class 1 is the page)
        probs = prediction_outputs['probs'][0][:, :, 1]
        assert probs.shape == (height, width)

        return probs, (width, height)

    def process(self, image):

        probs, (width, height) = self.get_raw_prediction(image)
        boxes = get_boxes(probs)

        return boxes, (width, height)

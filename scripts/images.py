import numpy as np
from PIL import Image, ImageDraw


def img_float_to_uint8(img):
    """ 2D float array between 0 and 1 is turned to 2D uint8."""
    if img.dtype == np.uint8:
        return img
    assert np.max(img) <= 1
    return (img * 255).astype(np.uint8)


def img_to_three_channels(img):
    """If the array has only one channel (shape H x W) copy it three times to have shape H x W x 3"""
    if len(img.shape) == 3:
        return img
    if len(img.shape) == 2:
        return np.array([img] * 3).transpose([1, 2, 0])
    raise ValueError(
        f"Image array has shape {img.shape}, can't convert to three channels.")


def img_to_PIL(img):
    """ 2D float array is turned to PIL image.
    Img can be float between 0 and 1 or uint8.
    """
    img = img_to_three_channels(img_float_to_uint8(img))
    img = Image.fromarray(img)
    return img


def concat_PIL_images(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def concat_PIL_images(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width + 1, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width + 1, 0))
    return dst


def resize_image(image_array, height, width):
    """Resize image (2D array).
    Array can be float between 0 and 1 or uint8."""
    image_array = img_to_PIL(image_array)
    image_array = image_array.resize((width, height))
    return np.asarray(image_array)[:, :, 0]


def draw_boxes_on_image(img, boxes, fill=None, outline=None, width=1):
    """Draws boxes [(x1,y1,x2,y2),...] on PIL image in-place."""
    draw = ImageDraw.Draw(img)
    for b in boxes:
        draw.rectangle(b, outline=outline, width=width, fill=fill)

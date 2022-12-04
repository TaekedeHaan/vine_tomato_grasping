"""
Created on Tue Jun 16 16:05:23 2020

@author: taeke
"""
import logging
import math
import typing


# External imports
import cv2
import numpy as np
import skimage


logger = logging.getLogger(__name__)


def add(image1, image2):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    """Add two images.

    Args:
        image1: The first image.
        image2: THe second image.

    Raises:
        ValueError: If the shape of both images do not match.

    Returns:
        The added images.
    """
    if not image1.shape == image2.shape:
        raise ValueError("Shape mismatch: cannot add image as the shape %s does not match with %s" %
                         (image1.shape, image2.shape))
    return cv2.bitwise_or(image1, image2)


def rotate(image, angle):
    # type: (np.ndarray, float) -> np.ndarray
    """ Construct an rotated imaged by angle from the provided image>

    Args:
        image: The image to rotate.
        angle: The angle by which to rotate the image in radians. 

    Returns:
        The rotated image
    """
    # Rotate a copy of the image, nd resize it such that no parts will be cut off.
    image_rotate = skimage.transform.rotate(image.copy(), math.degrees(angle), resize=True)

    # Skimage rotate returns the image as a float in range [0, 1], this needs to be converted to the original dtype.
    return (np.iinfo(image.dtype).max * image_rotate).astype(image.dtype, copy=False)


def crop(image, bounding_box):
    # type: (np.ndarray, typing.List[int]) -> np.ndarray
    """ Crop the image by the provided bounding box.

    Args:
        image: The original image
        bounding_box: The bounding box by which to crop the image [x, y, w, h].

    Returns:
        The cropped image.
    """
    x, y, w, h = bounding_box
    return image[y:y+h, x:x+w]


def compute_orientation(image):
    # type: (np.ndarray) -> float
    """ Compute the orientation in radians of the largest region in the image.

    Args:
        image: The image,

    Returns:
        The orientation in radians.
    """
    regions = skimage.measure.regionprops(skimage.measure.label(image), coordinates='xy')
    if len(regions) > 1:
        logger.warning("Multiple regions found while computing the orientation: the largest region will be used.")

    return typing.cast(float, regions[0].orientation)


def compute_bounding_box(image):
    # type: (np.ndarray) -> np.ndarray
    """ Find a bounding box around a provided image.

    Args:
        image: The image.

    Returns:
        The bounding box [x, y, w, h].
    """
    return cv2.boundingRect(image)

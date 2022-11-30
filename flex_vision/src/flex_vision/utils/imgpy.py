"""
Created on Tue Jun 16 16:05:23 2020

@author: taeke
"""
import logging

# External imports
import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.transform import rotate as ski_rotate

logger = logging.getLogger(__name__)


def check_dimensions(image1, image2):
    """check_dimensions"""
    return image1.shape == image2.shape


def add(image1, image2):
    """add two images"""
    if check_dimensions(image1, image2):
        return cv2.bitwise_or(image1, image2)
    else:
        raise ValueError("Cannot add images: its dimensions do not match")


def rotate(image, angle):
    """returns a new image, rotated by angle in radians
    angle: counter clockwise rotation in radians
    """
    dtype = image.dtype
    value_max = np.iinfo(dtype).max
    image_new = image.copy()

    # rotate returns a float in range [0, 1], this needs to be converted
    image_rotate = ski_rotate(image_new, np.rad2deg(angle), resize=True)
    image_rotate = (value_max * image_rotate).astype(dtype, copy=False)
    return image_rotate


def crop(image, angle, bounding_box):
    """returns a new image, rotated by angle in radians and cropped by the boundingbox"""
    image = rotate(image, angle)
    image = cut(image, bounding_box)
    return image


def cut(image, bounding_box):
    """returns the image cut, cut at the boundingbox"""
    x = bounding_box[0]
    y = bounding_box[1]
    w = bounding_box[2]
    h = bounding_box[3]
    return image[y:y+h, x:x+w]


def compute_angle(image):
    """returns the angle in radians based on the image"""
    regions = regionprops(label(image), coordinates='xy')

    if len(regions) > 1:
        logger.warning("Multiple regions found")

    return regions[0].orientation


def bbox(image):
    """find bounding box around a mask"""
    return cv2.boundingRect(image)

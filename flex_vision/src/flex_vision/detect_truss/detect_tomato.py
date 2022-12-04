"""
@author: taeke
"""

# External imports
import cv2
import logging
import numpy as np
from typing import TYPE_CHECKING

# Flex vision imports
from flex_vision.utils.util import plot_features

if TYPE_CHECKING:
    # pylint: disable=unused-import
    import typing

logger = logging.getLogger(__name__)


def compute_com(centers, radii):
    """
    Calculate the com of a set of spheres given their 2D centers and radii
    """
    centers = np.matrix(centers)
    radii = np.array(radii)
    return np.array((radii ** 3) * centers / (np.sum(radii ** 3)))


def detect_tomato(img_segment,     # type: typing.Optional[np.ndarray]
                  settings=None,   # type: typing.Optional[typing.Dict[str, typing.Any]]
                  px_per_mm=None,  # type: typing.Optional[float]
                  img_rgb=None,    # type: typing.Optional[np.ndarray]
                  save=False,      # type: bool
                  pwd="",          # type: str
                  name=""          # type: str
                  ):
    # type: (...) -> typing.Tuple(typing.Any, typing.Any, typing.Amy)
    """ Detect tomatoes in a provided image

    Args:
        img_segment: The segment in which to detect the tomatoes.
        settings: The detect tomato settings dictionary. Default to None.
        px_per_mm: The pixels per millimeter estimate. If provided the tomato sizes can be limited in millimeters. Default to None.
        img_rgb: The RGB image, used for plotting. Default to None.
        save: Save the image is set to True. Defaults to False.
        pwd: The path to store the saved image. Defaults to "".
        name: The file name to use when saving the image. Defaults to "".

    Returns:
        centers: The tomato center locations in the provided image.
        radii: The estimated tomato radii in the provided image.
        com: The estimated center of mass.
    """
    img_rgb = img_rgb if img_rgb is not None else img_segment
    settings = settings if settings is not None else settings.detect_tomato()

    # Set settings
    if px_per_mm:
        radius_min_px = int(round(px_per_mm * settings['radius_min_mm']))
        radius_max_px = int(round(px_per_mm * settings['radius_max_mm']))
    else:
        dim = img_segment.shape
        radius_min_px = dim[1] / settings['radius_min_frac']
        radius_max_px = dim[1] / settings['radius_max_frac']
    distance_min_px = radius_min_px * 2

    # Hough requires a gradient, thus the image is blurred
    blur_size = settings['blur_size']
    truss_blurred = cv2.GaussianBlur(img_segment, (blur_size, blur_size), 0)

    # fit circles: [x, y, radius]
    circles = cv2.HoughCircles(truss_blurred,
                               cv2.HOUGH_GRADIENT,
                               settings['dp'],
                               distance_min_px,
                               param1=settings['param1'],
                               param2=settings['param2'],
                               minRadius=radius_min_px,
                               maxRadius=radius_max_px)

    if circles is not None:
        centers, radii = circles[0][:, :2], circles[0][:, 2]  # [x, y, r]
        com = compute_com(centers, radii)
    else:
        centers, radii, com = [], [], None

    n_detected = len(radii)
    if save:
        tomato = {'centers': centers, 'radii': radii, 'com': com}
        plot_features(img_rgb, tomato=tomato, pwd=pwd, file_name=name + '_1', zoom=True)

    # Remove the circles which do not overlap with the tomato segment
    i_keep = find_overlapping_tomatoes(centers, radii, img_segment, ratio_threshold=settings['ratio_threshold'])
    if len(i_keep) != 0:
        centers = centers[i_keep, :]
        radii = radii[i_keep]
        com = compute_com(centers, radii)

    if len(i_keep) != n_detected:
        logger.info("Removed %d tomato(es) based on overlap", n_detected - len(i_keep))

    # visualize result
    if save:
        tomato = {'centers': centers, 'radii': radii, 'com': com}
        plot_features(img_rgb, tomato=tomato, pwd=pwd, file_name=name + '_2', zoom=True)

    return centers, radii, com


def find_overlapping_tomatoes(centers, radii, img_segment, ratio_threshold=0.5):
    iKeep = []
    N = centers.shape[0]
    for i in range(0, N):

        image_empty = np.zeros(img_segment.shape, dtype=np.uint8)
        mask = cv2.circle(image_empty, (centers[i, 0], centers[i, 1]), radii[i], 255, -1)

        res = cv2.bitwise_and(img_segment, mask)
        pixels = np.sum(res == 255)
        total = np.pi * radii[i] ** 2
        ratio = pixels / total
        if ratio > ratio_threshold:
            iKeep.append(i)

    return iKeep

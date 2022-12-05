""" detect_tomato.py: contains the main detect tomato procedure. """
# External imports
import cv2
import logging
import numpy as np
from typing import TYPE_CHECKING

# Flex vision imports
from flex_vision.detect_truss import settings as detect_truss_settings
from flex_vision.utils.util import plot_features

if TYPE_CHECKING:
    # pylint: disable=unused-import
    import typing

logger = logging.getLogger(__name__)


def detect_tomato(img_segment,     # type: np.ndarray
                  settings=None,   # type: typing.Optional[typing.Dict[str, typing.Any]]
                  px_per_mm=None,  # type: typing.Optional[float]
                  img_rgb=None,    # type: typing.Optional[np.ndarray]
                  save=False,      # type: bool
                  pwd="",          # type: str
                  name=""          # type: str
                  ):
    # type: (...) -> typing.Tuple[typing.List[typing.List[float]], typing.List[float], typing.Any]
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
    # Defaults
    img_rgb = img_rgb if img_rgb is not None else img_segment
    settings = settings if settings is not None else detect_truss_settings.detect_tomato()

    # Set settings
    if px_per_mm:
        radius_min_px = int(round(px_per_mm * settings['radius_min_mm']))
        radius_max_px = int(round(px_per_mm * settings['radius_max_mm']))
    else:
        dim = img_segment.shape
        radius_min_px = dim[1] / settings['radius_min_frac']
        radius_max_px = dim[1] / settings['radius_max_frac']

    # Hough requires a gradient, thus the image is blurred
    blur_size = settings['blur_size']
    truss_blurred = cv2.GaussianBlur(img_segment, (blur_size, blur_size), 0)

    # fit circles: [x, y, radius]
    circles = cv2.HoughCircles(truss_blurred,
                               cv2.HOUGH_GRADIENT,
                               settings['dp'],
                               radius_min_px * 2,
                               param1=settings['param1'],
                               param2=settings['param2'],
                               minRadius=radius_min_px,
                               maxRadius=radius_max_px)

    if circles is not None:
        # Centers consist of a list with [x, y, r], split into centers and radius
        centers = circles[0][:, :2].tolist()
        radii = circles[0][:, 2].tolist()
        com = compute_com(centers, radii)
    else:
        logger.warning(
            "Failed to detect any tomatoes using the cv2.HoughCircle detector within radius range [%d, %d]", radius_min_px, radius_max_px)
        centers, radii, com = [], [], None

    n_detected = len(radii)
    if save:
        tomato = {'centers': centers, 'radii': radii, 'com': com}
        plot_features(img_rgb, tomato=tomato, pwd=pwd, file_name=name + '_1', zoom=True)

    # Remove the circles which do not overlap with the tomato segment
    centers, radii = select_filled_circles(centers, radii, img_segment, ratio_threshold=settings['ratio_threshold'])
    if len(radii) != n_detected:
        com = compute_com(centers, radii)  # recompute CoM
        logger.info("Removed %d tomato(es) based on overlap", n_detected - len(radii))

    # visualize result
    if save:
        tomato = {'centers': centers, 'radii': radii, 'com': com}
        plot_features(img_rgb, tomato=tomato, pwd=pwd, file_name=name + '_2', zoom=True)

    return centers, radii, com


def select_filled_circles(centers,             # type: typing.List[typing.List[float]]
                          radii,               # type: typing.List[float]
                          mask,                # type: np.ndarray
                          ratio_threshold=0.5  # float
                          ):
    # type: (...) -> typing.Tuple[typing.List[typing.List[float]], typing.List[float]]
    """ Select circles which not overlap with the provided mask.

    Args:
        centers: The circle centers.
        radii: The circle radii.
        mask: The mask.
        ratio_threshold: The overlap ratio threshold. Defaults to 0.5.

    Returns:
        The selected circle centers.
        The selected circle radii.
    """
    centers_out = []
    radii_out = []
    for center, radius in zip(centers, radii):
        rounded_center = (int(round(center[0])), int(round(center[1])))
        empty_mask = np.zeros(mask.shape, dtype=np.uint8)
        circle_mask = cv2.circle(empty_mask, rounded_center, int(round(radius)), 255, -1)
        overlap_mask = cv2.bitwise_and(mask, circle_mask)
        overlap_ratio = (np.sum(overlap_mask == 255) / (np.pi * radius ** 2))
        if overlap_ratio < ratio_threshold:
            logger.debug("Removing circle %.1f, %.1f with radius %.1f because it only overlaps with %.1f%% with the mask",
                         center[0], center[1], radius, 100*overlap_ratio)
            continue

        centers_out.append(center)
        radii_out.append(radius)

    return centers_out, radii_out


def compute_com(centers, radii):
    # type: (typing.List[typing.List[float]], typing.List[float]) -> typing.List[float]
    """ Calculate the com of a set of spheres given their 2D centers and radii.

    Args:
        centers: List of 2D center coordinates.
        radii: List of sphere radii.

    Returns:
        The center of mass.
    """
    if not centers:
        raise ValueError("Empty input: cannot compute CoM for zero tomatoes")
    radii3 = np.array(radii) ** 3
    center_weighted = np.array([[radius3 * center[0], radius3 * center[1]] for center, radius3 in zip(centers, radii3)])
    return np.sum(center_weighted, axis=0)/sum(radii3).tolist()  # type: ignore

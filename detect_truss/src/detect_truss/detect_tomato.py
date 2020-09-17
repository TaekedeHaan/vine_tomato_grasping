#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 08:52:33 2020

@author: taeke
"""

## imports ##
import cv2
import numpy as np

# custom functions
from util import plot_features


def set_detect_tomato_settings(blur_size=3,
                               radius_min_frac=8,
                               radius_max_frac=4,
                               distance_min_frac=4,  # = tomato_radius_max
                               radius_min_mm=30,
                               radius_max_mm=40,
                               dp=4,
                               param1=20,
                               param2=80,
                               ratio_threshold=0.6):

    settings = {'radius_min_frac': radius_min_frac,
                'radius_max_frac': radius_max_frac,
                'distance_min_frac': distance_min_frac,
                'radius_min_mm': radius_min_mm,
                'radius_max_mm': radius_max_mm,
                'dp': dp,
                'param1': param1,
                'param2': param2,
                'ratio_threshold': ratio_threshold,
                'blur_size': blur_size}

    return settings


def detect_tomato(img_segment, settings, px_per_mm=None, img_rgb=None,
                  save=False, pwd="", name=""):
    if img_rgb is None:
        img_rgb = img_segment

    # set dimensions
    if px_per_mm:
        radius_min_px = int(round(px_per_mm * settings['radius_min_mm']))
        radius_max_px = int(round(px_per_mm * settings['radius_max_mm']))
        distance_min_px = radius_min_px * 2
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

    if circles is None:
        centers_overlap = None
        radii_overlap = None
        com_overlap = None

        centers = None
        radii = None
        com = None

    else:
        # swap columns [r, c] -> [x,y]
        centers_overlap = np.matrix(circles[0][:, 0:2])
        radii_overlap = circles[0][:, 2]
        com_overlap = (radii_overlap ** 3) * centers_overlap / (np.sum(radii_overlap ** 3))
        n_overlap = len(radii_overlap)

        # remove circles which do not overlapp with the tomato segment
        i_keep = find_overlapping_tomatoes(centers_overlap,
                                           radii_overlap,
                                           img_segment,
                                           ratio_threshold=settings['ratio_threshold'])

        n = len(i_keep)

        if n != n_overlap:
            print 'removed', n_overlap - n, 'tomaoto(es) based on overlap'

        if len(i_keep) == 0:
            centers = None
            radii = None
            com = None
        else:

            centers = centers_overlap[i_keep, :]
            radii = radii_overlap[i_keep]
            com = (radii ** 3) * centers / (np.sum(radii ** 3))

    # visualize result
    thickness = 1

    if save:
        tomato = {'centers': centers, 'radii': radii, 'com': com}
        tomato_overlap = {'centers': centers_overlap, 'radii': radii_overlap, 'com': com_overlap}
        plot_features(img_rgb, tomato=tomato_overlap, pwd=pwd, file_name=name + '_1', thickness=thickness)
        plot_features(img_rgb, tomato=tomato, pwd=pwd, file_name=name + '_2', thickness=thickness)

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


if __name__ == '__main__':
    print("This file has no main!")

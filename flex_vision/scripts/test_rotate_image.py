#!/usr/bin/env python2
"""
Created on Mon Feb 17 14:35:14 2020

@author: taeke
"""

import numpy as np
from matplotlib import pyplot as plt
from flex_vision.utils import geometry


def main():
    # %% INIT
    height = 80
    width = 120

    x = 40
    y = 10
    xy_original = [x, y]

    vmin = 0
    vmax = 255

    original_frame = 'original'
    local_frame = 'local'

    plt.close('all')

    for angle in np.linspace(-np.pi, np.pi, 40):
        transform = geometry.image_transform(original_frame, local_frame, [height, width], angle)
        point = geometry.Point2D(xy_original, original_frame)

        # create and rotate image
        img = np.full([height, width], 100, dtype=np.uint8)
        img[xy_original[1], xy_original[0]] = vmax
        img_rotate = img.copy().rotate(angle)

        point_new = geometry.Point2D(coord_r, local_frame)
        coord_r = transform.apply(point, local_frame).coord()
        coord_p = transform.apply(point_new, original_frame).coord()

        fig = plt.figure()
        plt.subplot(2, 2, 1)
        ax = fig.gca()
        plt.imshow(img, vmin=vmin, vmax=vmax)
        circle = plt.Circle([xy_original[0], xy_original[1]], 3, color='r', fill=False)
        ax.add_artist(circle)

        plt.subplot(2, 2, 2)
        ax = fig.gca()
        plt.imshow(img_rotate, vmin=vmin, vmax=vmax)
        circle = plt.Circle((coord_r[0], coord_r[1]), 3, color='r', fill=False)
        ax.add_artist(circle)

        plt.subplot(2, 2, 3)
        ax = fig.gca()
        plt.imshow(img, vmin=vmin, vmax=vmax)

        circle = plt.Circle((coord_p[0], coord_p[1]), 3, color='r', fill=False)
        ax.add_artist(circle)

        plt.show()


if __name__ == '__main__':
    main()

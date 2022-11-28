#!/usr/bin/env python2
"""
Created on Thu Jul 16 19:49:14 2020

@author: taeke
"""
import os

# External imports
import cv2

# Flex vision imports
from flex_vision import constants
from flex_vision.utils.util import change_brightness
from flex_vision.utils.util import create_directory, load_image
from flex_vision.detect_truss import settings
from flex_vision.detect_truss.process_image import ProcessImage
from flex_vision.detect_truss.detect_tomato import detect_tomato


def main():
    i_start = 3
    i_end = 4
    N = i_end - i_start + 1

    create_directory(constants.PATH_RESULTS)
    brightness = 0.85

    for count, i_tomato in enumerate(range(i_start, i_end)):
        print("Analyzing image %d out of %d" % (count + 1, N))

        tomatoID = str(i_tomato).zfill(3)
        tomato_name = tomatoID
        file_name = tomato_name + ".png"

        imRGB = load_image(os.path.join(constants.PATH_DATA, file_name), horizontal=True)

        image = ProcessImage(use_truss=True)

        image.add_image(imRGB)

        image.color_space()
        image.segment_image()
        image.filter_image()
        image.rotate_cut_img()

        # set parameters
        detect_tomato_settings = settings.detect_tomato()

        img_rgb = image.get_rgb(local=True)
        img_rgb_bright = change_brightness(img_rgb, brightness)

        image_tomato, image_peduncle, _ = image.get_segments(local=True)
        image_gray = cv2.bitwise_or(image_tomato, image_peduncle)

        centers, radii, com = detect_tomato(image_gray,
                                            detect_tomato_settings,
                                            img_rgb=img_rgb,
                                            save=True,
                                            pwd=constants.PATH_RESULTS,
                                            name=tomato_name)


if __name__ == '__main__':
    main()

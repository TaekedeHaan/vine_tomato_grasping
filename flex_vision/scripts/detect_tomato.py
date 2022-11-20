#!/usr/bin/env python2
"""
Created on Thu Jul 16 19:49:14 2020

@author: taeke
"""
import os

# External imports
import cv2

# Flex vision imports
from flex_vision.utils.util import change_brightness
from flex_vision.utils.util import make_dirs, load_rgb
from flex_vision.detect_truss import settings
from flex_vision.detect_truss.ProcessImage import ProcessImage
from flex_vision.detect_truss.detect_tomato import detect_tomato


def main():
    i_start = 1
    i_end = 22
    N = i_end - i_start + 1

    drive = "backup"  # "UBUNTU 16_0"  #
    pwd_root = os.path.join(os.sep, "media", "taeke", drive, "thesis_data",
                            "flex_vision")
    dataset = "lidl"

    pwd_data = os.path.join(pwd_root, "data", dataset)

    pwd_results = os.path.join(pwd_root, "results", dataset, "05_tomatoes")

    make_dirs(pwd_data)
    make_dirs(pwd_results)

    imMax = 255
    count = 0
    brightness = 0.85

    for count, i_tomato in enumerate(range(i_start, i_end + 1)):
        print("Analyzing image %d out of %d" % (count + 1, N))

        tomatoID = str(i_tomato).zfill(3)
        tomato_name = tomatoID
        file_name = tomato_name + ".png"

        imRGB = load_rgb(pwd_data, file_name, horizontal=True)

        image = ProcessImage(use_truss=True)

        image.add_image(imRGB)

        image.color_space()
        image.segment_image()
        image.filter_image()
        image.rotate_cut_img()

        # set parameters
        settings = settings.detect_tomato()

        img_rgb = image.get_rgb(local=True)
        img_rgb_bright = change_brightness(img_rgb, brightness)

        image_tomato, image_peduncle, _ = image.get_segments(local=True)
        image_gray = cv2.bitwise_or(image_tomato, image_peduncle)

        centers, radii, com = detect_tomato(image_gray,
                                            settings,
                                            img_rgb=img_rgb,
                                            save=True,
                                            pwd=pwd_results,
                                            name=tomato_name)


if __name__ == '__main__':
    main()

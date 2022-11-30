#!/usr/bin/env python2
"""
Created on Thu Jul 16 18:16:42 2020

@author: taeke
"""
import os

# Flex vision imports
from flex_vision.utils.util import create_directory, load_image
from flex_vision.detect_truss.process_image import ProcessImage


def main():
    nDigits = 3
    i_start = 1
    i_end = 2
    N = i_end - i_start

    save = True

    drive = "backup"  # "UBUNTU 16_0" #
    pwd_root = os.path.join(os.sep, "media", "taeke", drive, "thesis_data", "detect_truss")

    dataset = 'lidl'

    pwd_data = os.path.join(pwd_root, "data", dataset)
    pwd_results = os.path.join(pwd_root, "results", dataset, "06_peduncle")
    create_directory(pwd_results)

    for count, i_tomato in enumerate(range(i_start, i_end)):
        print("Analyzing image %d out of %d" % (i_tomato, N))

        tomato_name = str(i_tomato).zfill(nDigits)
        file_name = tomato_name + ".png"

        img_rgb = load_image(os.path.join(pwd_data, file_name), horizontal=True)

        image = ProcessImage(name=tomato_name,
                             pwd=pwd_results,
                             save=False)

        image.add_image(img_rgb)

        image.color_space()
        image.segment_image()
        image.filter_image()
        image.rotate_cut_img()
        image.detect_tomatoes()
        image.detect_peduncle()


if __name__ == '__main__':
    main()

#!/usr/bin/env python2
"""
Created on Fri May 22 12:04:59 2020

@author: taeke
"""
import os
import cv2
import json

# Flex vision imports
from flex_vision.utils.util import load_image
from flex_vision.utils.util import create_directory
from flex_vision.detect_truss.process_image import ProcessImage

# ls | cat -n | while read n f; do mv "$f" `printf "%03d.png" $n`; done
null = None


def initialize_label_me_dict(image_name, width, height):
    """initialize label me dictionary"""
    label_dict = {
        "shapes": [],
        "imagePath": image_name,
        "flags": {},
        "version": "4.5.6",
        "imageData": null,
        "imageWidth": width,
        "imageHeight": height
    }
    return label_dict


def initialize_kili_dict(job_name="JOB_0"):
    label_dict = {
        job_name: {
            "annotations": []
        }
    }
    return label_dict


def get_label_me_shape(contour, label):
    shape_dict = {
        "shape_type": "polygon",
        "points": contour,
        "flags": {},
        "group_id": null,
        "label": label.lower()
    }
    return shape_dict


def get_kili_shape(contour, label, width, height):
    width = float(width)
    height = float(height)

    coordinate_list = []
    for coordinate in contour:
        coordinate_dict = {"x": coordinate[0]/width, "y": coordinate[1]/height}
        coordinate_list.append(coordinate_dict)

    annotation = {
        "boundingPoly": [{
            "normalizedVertices": coordinate_list}
        ],
        "categories": [{"name": label.upper(), "confidence": 100}],
        "type": "polygon"
    }
    return annotation


def main():
    i_start = 1  # tomato file to load
    i_end = 85
    N = i_end - i_start

    extension = ".png"
    dataset = "lidl"  # "failures" #

    drive = "backup"  # "UBUNTU 16_0"  #
    pwd_root = os.path.join(os.sep, "media", "taeke", drive, "thesis_data", "detect_truss")
    pwd_data = os.path.join(pwd_root, "data", dataset)
    pwd_results = os.path.join(pwd_root, "labels", dataset)

    create_directory(pwd_results)
    process_image = ProcessImage()
    epsilon_factor = 0.0005  # contour simplification factor, lower is simpler!

    for count, i_tomato in enumerate(range(i_start, i_end)):
        print("Analyzing image ID %d (%d/%d)" % (i_tomato, count + 1, N))

        tomato_ID = str(i_tomato).zfill(3)
        tomato_name = tomato_ID
        file_name = tomato_name + "_rgb" + extension

        img_rgb = load_image(os.path.join(pwd_data, file_name), horizontal=True)
        process_image.add_image(img_rgb, name=tomato_name)
        process_image.color_space()
        process_image.segment_image()
        process_image.filter_image()

        tomato_label = 'TOMATO'
        stem_label = 'STEM'

        segments = {}
        segments[tomato_label] = process_image.tomato.data  # first segment is red
        segments[stem_label] = process_image.peduncle.data  # second segment is green

        width = img_rgb.shape[1]
        height = img_rgb.shape[0]
        label_me_dict = initialize_label_me_dict(tomato_name + extension, width=width, height=height)

        for label in segments:
            contours, _ = cv2.findContours(segments[label], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]

            for contour in contours:
                epsilon = epsilon_factor * cv2.arcLength(contour, True)
                polygon = cv2.approxPolyDP(contour, epsilon, True)
                polygon = polygon[:, 0].tolist()

                # Polygon must have more than two points
                if len(polygon) > 2:
                    label_me_dict["shapes"].append(get_label_me_shape(polygon, label))

        with open(os.path.join(pwd_results, tomato_ID + '.json'), 'w') as fp:
            json.dump(label_me_dict, fp)

        count = count + 1
        print("completed image %d out of %d" % (count, N))


if __name__ == '__main__':
    main()


# For cvat use the labelme2coco converter:
# python3 labelme2coco.py --noviz /media/taeke/backup/thesis_data/detect_truss/data/temp output --labels labels.txt

# to copy images:
# cp -r data/lidl/*.png labels/lidl/

# TODO:
# - file name should be $i.png
# - indexing should start at 1

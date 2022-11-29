#!/usr/bin/env python2
# External imports
import numpy as np
import unittest
from matplotlib import pyplot as plt
from typing import TYPE_CHECKING

# External imports
from flex_vision.utils import util
from flex_vision.detect_truss.process_image import ProcessImage
from flex_vision.analyze_results.analyze_results import index_true_positives


if TYPE_CHECKING:
    import typing


class TestProcessImage(unittest.TestCase):

    def test_process_image(self):
        process_image = ProcessImage()
        radius_minimum_mm = process_image.settings['detect_tomato']['radius_min_mm']
        radius_maximum_mm = process_image.settings['detect_tomato']['radius_max_mm']

        # generate truss
        px_per_mm = 3.75
        tomatoes, peduncle = generate_truss_features(
            [750, 350],
            radius_minimum_mm * px_per_mm,
            radius_maximum_mm * px_per_mm
        )

        # generate background image
        background_image = np.tile(np.array(util.background_color, ndmin=3, dtype=np.uint8), (1080, 1920, 1))

        # figsize = (background_image.shape[0]/20, background_image.shape[1]/20)
        # plt.figure(figsize=figsize, dpi=20)
        # plt.imshow(background_image)

        # plot truss
        util.plot_image(background_image)
        util.plot_truss(tomato=tomatoes, peduncle=peduncle)
        util.clear_axis()
        image = util.figure_to_image(plt.gcf())

        # process the generated image
        process_image.add_image(image, px_per_mm=px_per_mm, name='test')
        process_image.process_image()
        features_prediction = process_image.get_object_features()

        self.assertEqual(len(features_prediction['tomato']['centers']), 2)
        self.assertAlmostEqual(features_prediction['tomato']['centers'][0][0], 757.1025790183744)
        self.assertAlmostEqual(features_prediction['tomato']['centers'][0][1], 247.03665196688155)
        self.assertAlmostEqual(features_prediction['tomato']['centers'][1][0], 828.5812539598362)
        self.assertAlmostEqual(features_prediction['tomato']['centers'][1][1], 524.2131282930893)

        self.assertEqual(len(features_prediction['peduncle']['ends']), 2)
        self.assertAlmostEqual(features_prediction['peduncle']['ends'][0][0], 655.9700969384756)
        self.assertAlmostEqual(features_prediction['peduncle']['ends'][0][1], 460.9249898171696)
        self.assertAlmostEqual(features_prediction['peduncle']['ends'][1][0], 935.3054573725246)
        self.assertAlmostEqual(features_prediction['peduncle']['ends'][1][1], 292.38788289059414)

        self.assertEqual(len(features_prediction['peduncle']['junctions']), 2)
        self.assertAlmostEqual(features_prediction['peduncle']['junctions'][0][0], 768.514894424575)
        self.assertAlmostEqual(features_prediction['peduncle']['junctions'][0][1], 395.8814001344585)
        self.assertAlmostEqual(features_prediction['peduncle']['junctions'][1][0], 823.8057402629113)
        self.assertAlmostEqual(features_prediction['peduncle']['junctions'][1][1], 348.89286571933405)

        self.assertAlmostEqual(features_prediction['grasp_location']['xy'][0], 791.7370199859157)
        self.assertAlmostEqual(features_prediction['grasp_location']['xy'][1], 376.4717920906744)
        self.assertAlmostEqual(features_prediction['grasp_location']['angle'], -3.842480599372535)


def generate_truss_features(truss_center, radius_minimum, radius_maximum, angle_deg=30.0):
    # type (typing.List, float, float, float) -> typing.Tuple[typing.Dict, typing.Dict]
    """ Generate dictionarities containing tomato and peduncle features.

    Args:
        truss_center: truss center placement
        radii_range: Minimum and maximum radius
        angle: truss angle (in degree)
    """
    x = truss_center[0]
    y = truss_center[1]
    tomato_dist = 120

    angle_rad = np.deg2rad(angle_deg)
    rotation_matrix = np.array(((np.cos(angle_rad), -np.sin(angle_rad)), (np.sin(angle_rad), np.cos(angle_rad))))
    shift = 30

    # define truss features
    tomatoes = {'centers': [rotate_point([shift, -tomato_dist], rotation_matrix, truss_center),
                            rotate_point([-shift, tomato_dist], rotation_matrix, truss_center)],  # [x, y]
                'radii': [radius_minimum, radius_maximum]}

    peduncle = {'centers': [[x, y],
                            rotate_point([shift, -tomato_dist/2], rotation_matrix, truss_center),
                            rotate_point([-shift, tomato_dist/2], rotation_matrix, truss_center)],
                'angles': [-angle_rad, np.pi/2 - angle_rad, np.pi/2 - angle_rad],
                'length': [300, tomato_dist, tomato_dist],
                'junctions': [rotate_point([shift, 0], rotation_matrix, truss_center),
                              rotate_point([-shift, 0], rotation_matrix, truss_center)]}

    return tomatoes, peduncle


def rotate_point(point, rot_mat, transform):
    vec = np.matmul(np.array(point, ndmin=2), rot_mat) + np.array(transform, ndmin=2)
    return vec[0].tolist()


if __name__ == '__main__':
    unittest.main()

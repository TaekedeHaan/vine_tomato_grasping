#!/usr/bin/env python2
import os
import unittest
from typing import TYPE_CHECKING

# External imports
import numpy as np
from matplotlib import pyplot as plt

# External imports
from flex_vision.utils import util
from flex_vision.detect_truss.process_image import ProcessImage, load_px_per_mm
from flex_vision.analyze_results.analyze_results import index_true_positives, load_labels


if TYPE_CHECKING:
    # pylint: disable=unused-import
    import typing


class TestProcessImage(unittest.TestCase):

    def test_process_loaded_image(self):
        process_image = ProcessImage()
        path_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "doc", "data")
        image = util.load_image(os.path.join(path_data, "003.png"), horizontal=True)
        px_per_mm = load_px_per_mm(path_data, "003")
        tomato_labels, peduncle_labels = load_labels(os.path.join(path_data, "003.json"))

        # process the generated image
        process_image.add_image(image, px_per_mm=px_per_mm, name='test')
        process_image.process_image()
        features_prediction = process_image.get_object_features()

        self.assertEqual(len(features_prediction['tomato']['centers']), 5)
        self.assertEqual(len(features_prediction['peduncle']['ends']), 2)
        self.assertEqual(len(features_prediction['peduncle']['junctions']), 5)

        # analyze results
        i_prediction, i_label, false_pos, false_neg = index_true_positives(
            tomato_labels['centers'],
            features_prediction['tomato']['centers'],
            10,
            px_per_mm
        )

        # We expect two true positives, and zero false positives and negatives.
        self.assertEqual(len(i_prediction), 5)
        self.assertEqual(len(false_pos), 0)
        self.assertEqual(len(false_neg), 0)

        centers_prediction = np.array(features_prediction['tomato']['centers'])[i_prediction]
        radii_prediction = np.array(features_prediction['tomato']['radii'])[i_prediction]

        self.assertAlmostEqual(centers_prediction[0][0], 549.7881678698697)
        self.assertAlmostEqual(centers_prediction[0][1], 274.68872806054696)
        self.assertAlmostEqual(centers_prediction[1][0], 664.1974201115523)
        self.assertAlmostEqual(centers_prediction[1][1], 215.10246285203556)

        self.assertAlmostEqual(radii_prediction[0], 65.4000015258789)
        self.assertAlmostEqual(radii_prediction[1], 56.599998474121094)

        self.assertAlmostEqual(features_prediction['peduncle']['ends'][0][0], 563.8586837966153)
        self.assertAlmostEqual(features_prediction['peduncle']['ends'][0][1], 202.8902560604048)
        self.assertAlmostEqual(features_prediction['peduncle']['ends'][1][0], 729.2420385394588)
        self.assertAlmostEqual(features_prediction['peduncle']['ends'][1][1], 391.2790720076441)

        self.assertAlmostEqual(features_prediction['peduncle']['junctions'][0][0], 594.7981415781189)
        self.assertAlmostEqual(features_prediction['peduncle']['junctions'][0][1], 228.73111425409792)
        self.assertAlmostEqual(features_prediction['peduncle']['junctions'][1][0], 638.1158902406581)
        self.assertAlmostEqual(features_prediction['peduncle']['junctions'][1][1], 248.26499893710678)

        self.assertAlmostEqual(features_prediction['grasp_location']['xy'][0], 684.5548474015093)
        self.assertAlmostEqual(features_prediction['grasp_location']['xy'][1], 288.82881603822074)
        self.assertAlmostEqual(features_prediction['grasp_location']['angle'], 3.6575823053802727)
        self.assertEqual(features_prediction['grasp_location']['col'], 685)
        self.assertEqual(features_prediction['grasp_location']['row'], 289)

    def test_process_generated_image(self):
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

        dpi = 120
        figsize = (background_image.shape[1]/dpi, background_image.shape[0]/dpi)

        # plot truss
        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(background_image)
        util.plot_truss(tomato=tomatoes, peduncle=peduncle)
        util.clear_axis()
        image = util.figure_to_image(plt.gcf())

        # process the generated image
        process_image.add_image(image, px_per_mm=px_per_mm, name='test')
        process_image.process_image()
        features_prediction = process_image.get_object_features()

        self.assertEqual(len(features_prediction['tomato']['centers']), 2)
        self.assertEqual(len(features_prediction['peduncle']['ends']), 2)
        self.assertEqual(len(features_prediction['peduncle']['junctions']), 2)

        # analyze results
        i_prediction, i_label, false_pos, false_neg = index_true_positives(
            tomatoes['centers'],
            features_prediction['tomato']['centers'],
            10,
            px_per_mm
        )

        # We expect two true positives, and zero false positives and negatives.
        self.assertEqual(len(i_prediction), 2)
        self.assertEqual(len(false_pos), 0)
        self.assertEqual(len(false_neg), 0)

        centers_prediction = np.array(features_prediction['tomato']['centers'])[i_prediction]
        radii_prediction = np.array(features_prediction['tomato']['radii'])[i_prediction]

        self.assertAlmostEqual(centers_prediction[0][0], 712.4411785210777)
        self.assertAlmostEqual(centers_prediction[0][1], 236.37336778915875)
        self.assertAlmostEqual(centers_prediction[1][0], 783.4833019827396)
        self.assertAlmostEqual(centers_prediction[1][1], 471.47550877060104)

        self.assertAlmostEqual(radii_prediction[0], 116.59999847)
        self.assertAlmostEqual(radii_prediction[1], 147.3999939)

        self.assertAlmostEqual(features_prediction['peduncle']['ends'][0][0], 619.7435385686285)
        self.assertAlmostEqual(features_prediction['peduncle']['ends'][0][1], 425.93113615343884)
        self.assertAlmostEqual(features_prediction['peduncle']['ends'][1][0], 879.998726386269)
        self.assertAlmostEqual(features_prediction['peduncle']['ends'][1][1], 270.4536869397198)

        self.assertAlmostEqual(features_prediction['peduncle']['junctions'][0][0], 724.4119756581771)
        self.assertAlmostEqual(features_prediction['peduncle']['junctions'][0][1], 367.54005569391)
        self.assertAlmostEqual(features_prediction['peduncle']['junctions'][1][0], 774.8082578751593)
        self.assertAlmostEqual(features_prediction['peduncle']['junctions'][1][1], 333.8174410457796)

        self.assertAlmostEqual(features_prediction['grasp_location']['xy'][0], 749.9502396991849)
        self.assertAlmostEqual(features_prediction['grasp_location']['xy'][1], 350.312256363416)
        self.assertAlmostEqual(features_prediction['grasp_location']['angle'], -3.7234445123887827)


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

#!/usr/bin/env python2
import math
import unittest

# External imports
import cv2
import numpy as np

# Flex vision imports
from flex_vision.detect_truss import detect_tomato
from flex_vision.detect_truss import settings


class DetectTomatoTests(unittest.TestCase):

    def test_detect_tomato_none(self):

        # Add both circles
        circle_mask = np.zeros((200, 200), dtype=np.uint8)
        centers_out, radii_out, com_out = detect_tomato.detect_tomato(circle_mask)
        self.assertEqual(len(centers_out), 0)
        self.assertEqual(len(radii_out), 0)
        self.assertEqual(com_out, None)

        # Add both circles
        circle_mask = 255*np.ones((200, 200), dtype=np.uint8)
        centers_out, radii_out, com_out = detect_tomato.detect_tomato(circle_mask)
        self.assertEqual(len(centers_out), 0)
        self.assertEqual(len(radii_out), 0)
        self.assertEqual(com_out, None)

    def test_detect_tomato(self):
        centers = [[78.0, 155.0], [55.0, 103.0]]
        radii = [30.0, 25.0]

        my_settings = settings.detect_tomato(radius_min_frac=10, radius_max_frac=1)

        # Add both circles
        circle_mask = np.zeros((200, 200), dtype=np.uint8)
        circle_mask = cv2.circle(circle_mask, (int(centers[0][0]), int(centers[0][1])), int(radii[0]), 255, -1)
        circle_mask = cv2.circle(circle_mask, (int(centers[1][0]), int(centers[1][1])), int(radii[1]), 255, -1)
        centers_out, radii_out, com_out = detect_tomato.detect_tomato(circle_mask, settings=my_settings)
        self.assertEqual(len(centers_out), 2)
        self.assertEqual(len(radii_out), 2)
        np.testing.assert_almost_equal(centers_out, [[78.0, 154.0], [54.0, 102.0]])
        np.testing.assert_almost_equal(radii_out, [29.6000004, 23.2000008])
        self.assertAlmostEqual(com_out[0], 70.1998877)
        self.assertAlmostEqual(com_out[1], 137.0997568)

    def test_compute_com_no_tomato(self):
        centers = []
        radii = []
        self.assertRaises(ValueError, detect_tomato.compute_com, centers, radii)

    def test_compute_com_single_tomato(self):
        centers = [[10.0, 20.0]]
        radii = [10.0]
        com = detect_tomato.compute_com(centers, radii)
        self.assertAlmostEqual(com[0], 10.0)
        self.assertAlmostEqual(com[1], 20.0)

    def test_compute_com_identical_radius(self):
        centers = [[10.0, 20.0], [40.0, -20.0]]
        radii = [10.0, 10.0]
        com = detect_tomato.compute_com(centers, radii)
        self.assertAlmostEqual(com[0], 25.0)
        self.assertAlmostEqual(com[1], 0.0)

    def test_compute_com_different_radius(self):
        centers = [[10.0, 20.0], [40.0, -20.0]]
        radii = [10.0, 5.0]
        com = detect_tomato.compute_com(centers, radii)
        self.assertAlmostEqual(com[0], 13.3333333333333333)
        self.assertAlmostEqual(com[1], 15.555555555555556)

    def test_select_filled_circles_all_identical(self):
        centers = np.array([[10.0, 20.0], [40.0, 100.0]])
        radii = np.array([10.0, 5.0])

        # Add both circles
        circle_mask = np.zeros((200, 200), dtype=np.uint8)
        circle_mask = cv2.circle(circle_mask, (int(centers[0][0]), int(centers[0][1])), int(radii[0]), 255, -1)
        circle_mask = cv2.circle(circle_mask, (int(centers[1][0]), int(centers[1][1])), int(radii[1]), 255, -1)
        centers_out, radii_out = detect_tomato.select_filled_circles(centers, radii, circle_mask)
        self.assertEqual(len(centers_out), 2)
        self.assertEqual(len(radii_out), 2)
        np.testing.assert_almost_equal(centers_out, centers)
        np.testing.assert_almost_equal(radii_out, radii)

    def test_select_filled_circles_filter_missing(self):
        centers = np.array([[10.0, 20.0], [40.0, 100.0]])
        radii = np.array([10.0, 5.0])

        # Add both circles
        circle_mask = np.zeros((200, 200), dtype=np.uint8)
        circle_mask = cv2.circle(circle_mask, (int(centers[0][0]), int(centers[0][1])), int(radii[0]), 255, -1)
        centers_out, radii_out = detect_tomato.select_filled_circles(centers, radii, circle_mask)
        self.assertEqual(len(centers_out), 1)
        self.assertEqual(len(radii_out), 1)
        np.testing.assert_almost_equal(centers_out, [centers[0]])
        np.testing.assert_almost_equal(radii_out, [radii[0]])

    def test_select_filled_circles_smaller_radius(self):
        centers = np.array([[10.0, 20.0], [40.0, 100.0]])
        radii = np.array([10.0, 5.0])

        # Add both circles
        circle_mask = np.zeros((200, 200), dtype=np.uint8)
        circle_mask = cv2.circle(circle_mask, (int(centers[0][0]), int(centers[0][1])), 5, 255, -1)
        circle_mask = cv2.circle(circle_mask, (int(centers[1][0]), int(centers[1][1])), 2, 255, -1)
        centers_out, radii_out = detect_tomato.select_filled_circles(centers, radii, circle_mask)
        self.assertEqual(len(centers_out), 0)
        self.assertEqual(len(radii_out), 0)

    def test_select_filled_circles_larger_radius(self):
        centers = np.array([[50.0, 60.0], [140.0, 150.0]])
        radii = np.array([10.0, 5.0])

        # Add both circles
        circle_mask = np.zeros((200, 200), dtype=np.uint8)
        circle_mask = cv2.circle(circle_mask, (int(centers[0][0]), int(centers[0][1])), 30, 255, -1)
        circle_mask = cv2.circle(circle_mask, (int(centers[1][0]), int(centers[1][1])), 15, 255, -1)
        centers_out, radii_out = detect_tomato.select_filled_circles(centers, radii, circle_mask)
        self.assertEqual(len(centers_out), 2)
        self.assertEqual(len(radii_out), 2)
        np.testing.assert_almost_equal(centers_out, centers)
        np.testing.assert_almost_equal(radii_out, radii)

    def test_select_filled_circles_center_error(self):
        centers = np.array([[50.0, 60.0], [140.0, 150.0]])
        radii = np.array([10.0, 5.0])

        # Add both circles
        circle_mask = np.zeros((200, 200), dtype=np.uint8)
        circle_mask = cv2.circle(circle_mask, (int(centers[0][0]) + 5, int(centers[0][1]) + 5), int(radii[0]), 255, -1)
        circle_mask = cv2.circle(circle_mask, (int(centers[1][0]) + 3, int(centers[1][1]) + 3), int(radii[1]), 255, -1)
        centers_out, radii_out = detect_tomato.select_filled_circles(centers, radii, circle_mask)
        self.assertEqual(len(centers_out), 1)
        self.assertEqual(len(radii_out), 1)
        np.testing.assert_almost_equal(centers_out, [centers[0]])
        np.testing.assert_almost_equal(radii_out, [radii[0]])


if __name__ == '__main__':
    unittest.main()

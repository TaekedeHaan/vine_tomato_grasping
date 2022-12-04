#!/usr/bin/env python2
import math
import unittest

# External imports
import numpy as np

# Flex vision imports
from flex_vision.utils import imgpy


class ImgpyTests(unittest.TestCase):

    def test_rotate(self):
        image = np.eye(4, dtype=np.uint8)
        image_rotated = imgpy.rotate(image, math.radians(90))
        self.assertTupleEqual(image_rotated.shape, (4, 4))

        # TODO: the pixel at the 2nd row is lost due to rounding, look into wether this desired is desirable.
        image_rotated_expected = np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_almost_equal(image_rotated, image_rotated_expected)

        image = np.eye(4, dtype=np.uint8)
        image_rotated = imgpy.rotate(image, math.radians(45))
        self.assertTupleEqual(image_rotated.shape, (6, 6))
        np.testing.assert_array_almost_equal(image_rotated, np.zeros((6, 6)))

    def test_crop(self):
        image = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)
        image_cropped = imgpy.crop(image, (2, 1, 1, 1))
        self.assertTupleEqual(image_cropped.shape, (1, 1))
        self.assertEqual(image_cropped[0, 0], 1)

    def test_compute_orientation(self):
        image = np.eye(4, dtype=np.uint8)
        angle = imgpy.compute_orientation(image)
        self.assertAlmostEqual(angle, -math.radians(45))

    def test_compute_bounding_box(self):
        image = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)
        x, y, w, h = imgpy.compute_bounding_box(image)
        self.assertEqual(x, 2)
        self.assertEqual(y, 1)
        self.assertEqual(w, 1)
        self.assertEqual(h, 1)


if __name__ == '__main__':
    unittest.main()

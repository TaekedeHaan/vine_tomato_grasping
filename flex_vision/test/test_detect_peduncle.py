#!/usr/bin/env python2
import unittest

# External imports
import numpy as np

# External imports
from flex_vision.detect_truss.detect_peduncle import detect_peduncle


class TestDetectPeduncle(unittest.TestCase):

    def test_detect_peduncle_empty(self):
        shape = (720, 1280)
        image = np.zeros(shape, dtype=np.uint8)
        # mask, branch_data, junc_coords, end_coords = detect_peduncle(image)
        # self.assertTupleEqual(mask.shape, shape)
        # self.assertTupleEqual(junc_coords.shape, (0, 2))
        # self.assertTupleEqual(end_coords.shape, (0, 2))

    def test_detect_peduncle_fill(self):
        shape = (720, 1280)
        image = 255 * np.ones(shape, dtype=np.uint8)
        mask, branch_data, junc_coords, end_coords = detect_peduncle(image)

        self.assertTupleEqual(mask.shape, shape)
        self.assertTupleEqual(junc_coords.shape, (0, 2))

        self.assertTupleEqual(end_coords.shape, (2, 2))
        self.assertEqual(end_coords[0, 0], 359.0)
        self.assertEqual(end_coords[0, 1], 360.0)
        self.assertEqual(end_coords[1, 0], 920.0)
        self.assertEqual(end_coords[1, 1], 359.0)

    def test_detect_peduncle_1(self):
        shape = (720, 1280)
        image = np.zeros(shape, dtype=np.uint8)
        image[:, 200] = 255
        image[:, 500] = 255
        image[300, 10:-10] = 255
        mask, _, junc_coords, end_coords = detect_peduncle(image)

        self.assertTupleEqual(mask.shape, shape)
        self.assertTupleEqual(junc_coords.shape, (0, 2))

        self.assertTupleEqual(end_coords.shape, (2, 2))
        self.assertEqual(end_coords[0, 0], 1268.0)  # Would expect 1270
        self.assertEqual(end_coords[0, 1], 300.0)
        self.assertEqual(end_coords[1, 0], 500.0)  # Would expect 0
        self.assertEqual(end_coords[1, 1], 300.0)

    def test_detect_peduncle_eye(self):
        image = np.eye(1280, dtype=np.uint8)
        mask, branch_data, junc_coords, end_coords = detect_peduncle(image)

        self.assertTupleEqual(mask.shape, (1280, 1280))
        self.assertTupleEqual(junc_coords.shape, (0, 2))
        self.assertTupleEqual(end_coords.shape, (0, 2))  # Would expect (2, 2)


if __name__ == '__main__':
    unittest.main()

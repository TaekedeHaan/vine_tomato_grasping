#!/usr/bin/env python2
import unittest

# External imports
import numpy as np

# Flex vision imports
from flex_vision.utils import geometry


class TransformTests(unittest.TestCase):

    def test_translation(self):
        transform = geometry.Transform('origin', 'local', tx=3.0, ty=4.0)

        point1 = geometry.Point2D([0, 0], 'origin')
        point2 = geometry.Point2D([0, 0], 'local')

        self.asset_almost_equal_point2d(transform.apply(point1, 'origin'), geometry.Point2D([0, 0], 'origin'))
        self.asset_almost_equal_point2d(transform.apply(point1, 'local'), geometry.Point2D([-3, -4], 'local'))

        self.asset_almost_equal_point2d(transform.apply(point2, 'origin'), geometry.Point2D([3, 4], 'origin'))
        self.asset_almost_equal_point2d(transform.apply(point2, 'local'), geometry.Point2D([0, 0], 'local'))

        np.testing.assert_almost_equal(
            geometry.distance(transform.apply(point1, 'origin'), transform.apply(point2, 'origin')), 5.0)

    def test_90deg_rotation(self):

        shape = (10, 20)  # [height, width]
        angle = np.pi/2
        transform = geometry.image_transform('origin', 'local', shape, angle)

        point1 = geometry.Point2D([10, 5], 'origin')
        point2 = geometry.Point2D([5, 10], 'local')

        self.asset_almost_equal_point2d(transform.apply(point1, 'origin'), geometry.Point2D([10, 5], 'origin'))
        self.asset_almost_equal_point2d(transform.apply(point1, 'local'), geometry.Point2D([5, 10], 'local'))

        self.asset_almost_equal_point2d(transform.apply(point2, 'origin'), geometry.Point2D([10, 5], 'origin'))
        self.asset_almost_equal_point2d(transform.apply(point2, 'local'), geometry.Point2D([5, 10], 'local'))

        np.testing.assert_almost_equal(
            geometry.distance(transform.apply(point1, 'origin'), transform.apply(point2, 'origin')), 0.0)

    def test_45deg_rotation(self):

        shape = (20, 20)  # (height, width)
        angle = np.pi/4
        transform = geometry.image_transform('origin', 'local', shape, angle)

        point1 = geometry.Point2D([10, 10], 'origin')
        point2 = geometry.Point2D([np.sqrt(2)*10, np.sqrt(2)*10], 'local')

        self.asset_almost_equal_point2d(transform.apply(point1, 'origin'), geometry.Point2D([10, 10], 'origin'))
        self.asset_almost_equal_point2d(transform.apply(point1, 'local'),
                                        geometry.Point2D([np.sqrt(2)*10, np.sqrt(2)*10], 'local'))

        self.asset_almost_equal_point2d(transform.apply(point2, 'origin'), geometry.Point2D([10, 10], 'origin'))
        self.asset_almost_equal_point2d(transform.apply(point2, 'local'),
                                        geometry.Point2D([np.sqrt(2)*10, np.sqrt(2)*10], 'local'))

        np.testing.assert_almost_equal(
            geometry.distance(transform.apply(point1, 'origin'), transform.apply(point2, 'origin')), 0.0)

    def test_345_rotation(self):

        shape = (6, 8)  # (height, width)
        angle = -np.arctan2(3, 4)
        transform = geometry.image_transform('origin', 'local', shape, angle)

        point1 = geometry.Point2D([4, 3], 'origin')
        point2 = geometry.Point2D([5, 3.0/5.0*8], 'local')

        self.asset_almost_equal_point2d(transform.apply(point1, 'origin'), geometry.Point2D([4, 3], 'origin'))
        self.asset_almost_equal_point2d(transform.apply(point1, 'local'), geometry.Point2D([5, 3.0/5.0*8], 'local'))

        self.asset_almost_equal_point2d(transform.apply(point2, 'origin'), geometry.Point2D([4, 3], 'origin'))
        self.asset_almost_equal_point2d(transform.apply(point2, 'local'), geometry.Point2D([5, 3.0/5.0*8], 'local'))

        np.testing.assert_almost_equal(
            geometry.distance(transform.apply(point1, 'origin'), transform.apply(point2, 'origin')), 0.0)

    def test_transform_forwards_backwards(self):
        """
        get coordinate return the original coordinate after translating forwards and backwards
        """
        shape = [1000, 400]  # [width, height]
        tx, ty = (50, -10)
        for angle in np.arange(-np.pi, np.pi, 10):

            transform = geometry.image_transform('origin', 'local', shape, angle, tx=tx, ty=ty)
            point1 = geometry.Point2D([400, 100], 'origin')

            point2 = transform.apply(point1, 'local')

            self.asset_almost_equal_point2d(transform.apply(point1, 'origin'),
                                            transform.apply(point2, 'origin'))

    def test_missing_transform(self):
        """
        get coordinate returns a geometry.LookupException when requested a coordinate in a frame for which the transform is unknown
        """
        transform = geometry.Transform('origin', 'local')
        point1 = geometry.Point2D([400, 100], 'origin')
        self.assertRaises(geometry.LookupException, transform.apply, point1, 'space')

    def test_point_length_mismatch(self):
        """
        geometry.Point2D returns a ValueError when a wrong coordinate length is provided
        """
        self.assertRaises(ValueError, geometry.Point2D, [400, 100, 100], 'origin')

    def asset_almost_equal_point2d(self, point1, point2):
        # type: (geometry.Point2D, geometry.Point2D) -> None
        np.testing.assert_almost_equal(point1.coord, point2.coord)
        self.assertEqual(point1.frame_id, point2.frame_id)


if __name__ == '__main__':
    unittest.main()

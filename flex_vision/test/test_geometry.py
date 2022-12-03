#!/usr/bin/env python2
import unittest

# External imports
import numpy as np

# Flex vision imports
from flex_vision.utils import geometry


class TransformTests(unittest.TestCase):

    def test_translation(self):
        transform = geometry.Transform('origin', 'local', translation=[3, 4])

        point1 = geometry.Point2D([0, 0], 'origin')
        point2 = geometry.Point2D([0, 0], 'local')

        self.asset_almost_equal_point2d(transform.apply(point1, 'origin'), geometry.Point2D([0, 0], 'origin'))
        self.asset_almost_equal_point2d(transform.apply(point1, 'local'), geometry.Point2D([-3, -4], 'local'))

        self.asset_almost_equal_point2d(transform.apply(point2, 'origin'), geometry.Point2D([3, 4], 'origin'))
        self.asset_almost_equal_point2d(transform.apply(point2, 'local'), geometry.Point2D([0, 0], 'local'))

        # np.testing.assert_almost_equal(geometry.distance(point1, point2), 5)

    def test_90deg_rotation(self):

        shape = [20, 10]  # [width, height]
        angle = np.pi/2
        transform = geometry.Transform('origin', 'local', dim=shape, angle=angle)

        point1 = geometry.Point2D([10, 5], 'origin')
        point2 = geometry.Point2D([5, 10], 'local')

        self.asset_almost_equal_point2d(transform.apply(point1, 'origin'), geometry.Point2D([10, 5], 'origin'))
        self.asset_almost_equal_point2d(transform.apply(point1, 'local'), geometry.Point2D([5, 10], 'local'))

        self.asset_almost_equal_point2d(transform.apply(point2, 'origin'), geometry.Point2D([10, 5], 'origin'))
        self.asset_almost_equal_point2d(transform.apply(point2, 'local'), geometry.Point2D([5, 10], 'local'))

        # np.testing.assert_almost_equal(geometry.distance(point1, point2), 0)

    def test_45deg_rotation(self):

        shape = [20, 20]  # [width, height]
        angle = np.pi/4
        transform = geometry.Transform('origin', 'local', dim=shape, angle=angle)

        point1 = geometry.Point2D([10, 10], 'origin')
        point2 = geometry.Point2D([np.sqrt(2)*10, np.sqrt(2)*10], 'local')

        self.asset_almost_equal_point2d(transform.apply(point1, 'origin'), geometry.Point2D([10, 10], 'origin'))
        self.asset_almost_equal_point2d(transform.apply(point1, 'local'),
                                        geometry.Point2D([np.sqrt(2)*10, np.sqrt(2)*10], 'local'))

        self.asset_almost_equal_point2d(transform.apply(point2, 'origin'), geometry.Point2D([10, 10], 'origin'))
        self.asset_almost_equal_point2d(transform.apply(point2, 'local'),
                                        geometry.Point2D([np.sqrt(2)*10, np.sqrt(2)*10], 'local'))

        # np.testing.assert_almost_equal(geometry.distance(point1, point2), 0)

    def test_345_rotation(self):

        shape = [8, 6]  # [width, height]
        angle = -np.arctan2(3, 4)
        transform = geometry.Transform('origin', 'local', dim=shape, angle=angle)

        point1 = geometry.Point2D([4, 3], 'origin')
        point2 = geometry.Point2D([5, 3.0/5.0*8], 'local')

        self.asset_almost_equal_point2d(transform.apply(point1, 'origin'), geometry.Point2D([4, 3], 'origin'))
        self.asset_almost_equal_point2d(transform.apply(point1, 'local'), geometry.Point2D([5, 3.0/5.0*8], 'local'))

        self.asset_almost_equal_point2d(transform.apply(point2, 'origin'), geometry.Point2D([4, 3], 'origin'))
        self.asset_almost_equal_point2d(transform.apply(point2, 'local'), geometry.Point2D([5, 3.0/5.0*8], 'local'))

        # np.testing.assert_almost_equal(geometry.distance(point1, point2), 0)

    def test_transform_forwards_backwards(self):
        """
        get coordinate return the original coordinate after translating forwards and backwards
        """
        shape = [1000, 400]  # [width, height]
        translation = [50, -10]
        for angle_rad in np.arange(-np.pi, np.pi, 10):

            transform = geometry.Transform('origin', 'local', shape, angle_rad, translation)
            point1 = geometry.Point2D([400, 100], 'origin')

            point2 = transform.apply(point1, 'local')

            self.asset_almost_equal_point2d(transform.apply(point1, 'origin'),
                                            transform.apply(point2, 'origin'))

    def test_missing_transform(self):
        """
        get coordinate returns a geometry.LookupException when requested a coordinate in a frame for which the transform is unknown
        """
        transform = geometry.Transform('origin', 'local', translation=[0, 0])
        point1 = geometry.Point2D([400, 100], 'origin')
        self.assertRaises(geometry.LookupException, transform.apply, point1, 'space')

    def test_point_length_mismatch(self):
        """
        geometry.Point2D returns a ValueError when a wrong coordinate length is provided
        """
        self.assertRaises(ValueError, geometry.Point2D, [400, 100, 100], 'origin')

    def test_transform_length_mismatch(self):
        """
        geometry.Transform returns a ValueError when a wrong translation length is provided
        """
        self.assertRaises(ValueError, geometry.Transform, 'origin', 'local', translation=[0, 0, 0])

    def asset_almost_equal_point2d(self, point1, point2):
        # type: (geometry.Point2D, geometry.Point2D) -> None
        np.testing.assert_almost_equal(point1.coord, point2.coord)
        self.assertEqual(point1.frame_id, point2.frame_id)


if __name__ == '__main__':
    unittest.main()

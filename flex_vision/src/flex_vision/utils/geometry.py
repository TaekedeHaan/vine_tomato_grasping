import logging
import math
from typing import TYPE_CHECKING

# External imports
import numpy as np
from flex_vision import constants  # pylint: disable=unused-import

if TYPE_CHECKING:
    # pylint: disable=unused-import
    import typing

logger = logging.getLogger("flex_vision")


class Point2D(object):
    """ Class used for storing two-dimensional coordinate with respect to a certain reference frame """

    def __init__(self, coord, frame_id):
        # type: (typing.Any, str) -> None
        """
        coord: two-dimensional coordinates as [x, y]
        frame_id: the name of the frame
        """
        self._coord = vectorize(coord)
        self.frame_id = frame_id

    @property
    def coord(self):
        # type: () -> typing.List[float]
        return self._coord[:, 0].tolist()  # type: ignore


class Transform(object):
    """ Class used for storing a two-dimensional transformation, and applying it to two-dimensional points """

    def __init__(self, source_frame, target_frame, angle=0.0, tx=0.0, ty=0.0):
        # type: (str, str, float, float, float) -> None
        """ Con"""

        self.source_frame = source_frame
        self.target_frame = target_frame
        self.R = rotation_matrix(angle)
        self.T = vectorize([tx, ty])

    def apply(self, point, frame_id):
        # type: (Point2D, str) -> Point2D
        """
        Applies transform to a given point to a given frame
        point: Point object
        frame_id: string, name of frame id
        """
        if point.frame_id == frame_id:
            return point
        if point.frame_id == self.source_frame and frame_id == self.target_frame:
            return Point2D(self._forwards(vectorize(point.coord)), frame_id)
        elif point.frame_id == self.target_frame and frame_id == self.source_frame:
            return Point2D(self._backwards(vectorize(point.coord)), frame_id)
        else:
            raise LookupException(
                "Lookup error: can not transform point from %s to %s as the transform is still empty" % (point.frame_id, frame_id))

    def _forwards(self, coord):
        # type: (np.typing.ArrayLike) -> np.typing.ArrayLike
        """
        translates 2d coordinate with and angle and than translation
        coord: 2D coords [x, y]
        """
        return np.matmul(self.R.T, coord) - self.T

    def _backwards(self, coord):
        # type: (np.typing.ArrayLike) -> np.typing.ArrayLike
        """
        translates 2d coordiante with -translation and than -angle
        coord: 2D coords [x, y]
        """
        return np.matmul(self.R, coord + self.T)


def distance(point1, point2):
    # type: (Point2D, Point2D) -> float
    if point1.frame_id != point2.frame_id:
        raise ValueError("Frame mismatch: cannot compute distance between points, as they are defined in different frames")
    return math.sqrt(np.sum(np.power(np.subtract(point1.coord, point2.coord), 2)))


def image_transform(source_frame, target_frame, shape, angle, tx=0.0, ty=0.0):
    # type: (str, str, typing.Tuple[int, int], float, float, float) -> Transform
    """
    Args:
        source_frame: transform from frame name
        target_frame: transform to frame name
        shape: The image shape.
        angle: angle of rotation in radians (counter clockwise is positive)
        translation: translation [x,y]
    """
    height, width = shape

    # Translation vector due to rotation
    if angle > 0:
        tx_rotation, ty_rotation = (0.0, -np.sin(angle) * width)
    else:
        tx_rotation, ty_rotation = (np.sin(angle) * height, 0.0)

    # Translation vector due to > 90deg rotations
    if (angle < -np.pi/2) or (angle > np.pi/2):
        tx_rotation, ty_rotation = (tx_rotation + np.cos(angle) * width, ty_rotation + np.cos(angle) * height)

    return Transform(source_frame, target_frame, angle, tx_rotation + tx, ty_rotation + ty)


class LookupException(Exception):
    """Exception raised for failed to lookup a transform."""
    pass


def points_from_coords(coords, frame):
    "Takes a list of coordinates, and outputs a list of Point2D"
    return [Point2D(coord, frame) for coord in coords]


def coords_from_points(transform, points, frame):
    """Takes a list of points, and outputs a list of coordinates"""
    return [transform.apply(point, frame).coord for point in points]


def vectorize(data):
    """Takes a list, tuple or numpy array and returns a column vector"""
    if isinstance(data, (list, tuple)):
        # logger.info("list/tuple")
        if len(data) != 2:
            raise ValueError("Length mismatch: Expected list or tuple has 2 elements, but has %d elements" % len(data))
        return np.array(data, ndmin=2).transpose()

    elif isinstance(data, np.ndarray):
        # logger.info("numpy array")
        coord = np.array(data, ndmin=2)
        if coord.shape == (1, 2):
            return coord.transpose()
        elif coord.shape == (2, 1):
            return coord
        else:
            raise ValueError("Shape mismatch: Expected numpy array has shape (1, 2) or (2,1), but has %s" % data.shape)


def rotation_matrix(angle):
    # type: (float) -> np.typing.ArrayLike
    """Construct a 2D rotation matrix from a provided angle.

    Args:
        angle: The angle in radians.

    Returns:
        The rotation matrix
    """
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])


def main():
    """
    Brief demo of the Point2d and Transform class
    """
    transform = Transform('origin', 'local', translation=(3, 4))

    frame_id = 'origin'
    coord = [0, 0]
    point1 = Point2D(coord, frame_id)

    frame_id = 'local'
    coord = [0, 0]
    point2 = Point2D(coord, frame_id)

    for frame_id in ['origin', 'local']:
        logger.info("point 1 in %s: %s", frame_id, transform.apply(point1, frame_id).coord)
        logger.info("point 2 in %s: %s", frame_id, transform.apply(point2, frame_id).coord)

    point1_local = transform.apply(point1, 'local')
    point2_local = transform.apply(point2, 'local')
    logger.info("Distance between points: %.3f", distance(point1_local, point2_local))
    # print('distance between points: ', point1.dist(point2))


if __name__ == '__main__':
    main()

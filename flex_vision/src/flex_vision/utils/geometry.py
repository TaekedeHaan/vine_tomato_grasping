""" geometry.py: contains several classes and functions for tracking points in 2D. """
import logging
import math
from typing import TYPE_CHECKING

# External imports
import numpy as np

if TYPE_CHECKING:
    # pylint: disable=unused-import
    import typing

logger = logging.getLogger(__name__)


class Point2D(object):
    """ Class used for storing two-dimensional coordinate with respect to a certain reference frame """

    def __init__(self, coord, frame_id):
        # type: (typing.Any, str) -> None
        """ Construct a Point2D object.

        Args:
            coord: two-dimensional coordinates as [x, y].
            frame_id: The frame ID.
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
        """ Construct a Transform object

        Args:
            source_frame: The source frame where to transform from.
            target_frame: The target frame where to transform to.
            angle: The angle in radians. Defaults to 0.0.
            tx: The translation in x direction. Defaults to 0.0.
            ty: The translation in y direction. Defaults to 0.0.
        """
        self.source_frame = source_frame
        self.target_frame = target_frame
        self.R = rotation_matrix(angle)
        self.T = vectorize([tx, ty])

    def apply(self, point, frame_id):
        # type: (Point2D, str) -> Point2D
        """ Apply the transform to a given point to a given frame.

        Args:
            point: The point to transform.
            frame_id: The frame ID to transform to.
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
        """ Apply the transform forwards to the provided coordinate.

        Args:
            coord: The coordinate to which to apply the transform.

        Returns:
            The transformed coordinate.
        """
        return np.matmul(self.R.T, coord) - self.T

    def _backwards(self, coord):
        # type: (np.typing.ArrayLike) -> np.typing.ArrayLike
        """ Apply the transform backwards to the provided coordinate.

        Args:
            coord: The coordinate to which to apply the transform.

        Returns:
            The transformed coordinate.
        """
        return np.matmul(self.R, coord + self.T)


def distance(point1, point2):
    # type: (Point2D, Point2D) -> float
    """Compute the euclidean distance between two provided 2D-points.

    Args:
        point1: The first point.
        point2: The second point.

    Raises:
        ValueError: If the points are defined with respect to different frames. 

    Returns:
        The Euclidean distance.
    """
    if point1.frame_id != point2.frame_id:
        raise ValueError("Frame mismatch: cannot compute distance between points, as they are defined in different frames")
    return math.sqrt(np.sum(np.power(np.subtract(point1.coord, point2.coord), 2)))


def image_transform(source_frame, target_frame, shape, angle, tx=0.0, ty=0.0):
    # type: (str, str, typing.Tuple[int, int], float, float, float) -> Transform
    """ Compute the transform which results from cropping and rotating an image.

    Args:
        source_frame: transform from frame name
        target_frame: transform to frame name
        shape: The image shape.
        angle: angle of rotation in radians (counter clockwise is positive).
        tx: translation in x-direction. Defaults to 0.0.
        ty: translation in y-direction. Defaults to 0.0. 
    """
    height, width = shape

    # Translation due to rotation
    if angle > 0:
        tx_rotation, ty_rotation = (0.0, -np.sin(angle) * width)
    else:
        tx_rotation, ty_rotation = (np.sin(angle) * height, 0.0)

    # Additional translation vector due to > 90deg rotation
    if (angle < -np.pi/2) or (angle > np.pi/2):
        tx_rotation += np.cos(angle) * width
        ty_rotation += np.cos(angle) * height

    return Transform(source_frame, target_frame, angle, tx_rotation + tx, ty_rotation + ty)


class LookupException(Exception):
    """Exception raised for failed to lookup a transform."""
    pass


def points_from_coords(coords, frame):
    # type: (typing.Any, str) -> typing.List[Point2D]
    """ Construct a list of 2D points.

    Args:
        coords: The coordinates.
        frame: The frame.

    Returns:
        List of 2D points.
    """
    return [Point2D(coord, frame) for coord in coords]


def coords_from_points(points, transform, frame):
    # type: (typing.List[Point2D], Transform, str) -> typing.List[typing.List[float]]
    """ Constructs a list of coordinates with respect to a certain frame from a list of 2D points.

    Args:
        points: The list of 2D points.
        transform: The transform.
        frame: The target frame.

    Returns:
        A list of coordinates
    """
    return [transform.apply(point, frame).coord for point in points]


def vectorize(data):
    # type: (typing.Any) -> np.typing.ArrayLike
    """ Takes a list, tuple or numpy array and returns a column vector """
    if isinstance(data, (list, tuple)):
        if len(data) != 2:
            raise ValueError("Length mismatch: Expected list or tuple has 2 elements, but has %d elements" % len(data))
        return np.array(data, ndmin=2).transpose()

    elif isinstance(data, np.ndarray):
        coord = np.array(data, ndmin=2)
        if coord.shape == (1, 2):
            return coord.transpose()
        elif coord.shape == (2, 1):
            return coord
        else:
            raise ValueError(
                "Shape mismatch: Expected numpy array has shape (1, 2) or (2,1), but has shape %s" % (data.shape, ))


def rotation_matrix(angle):
    # type: (float) -> np.typing.ArrayLike
    """ Construct a 2D rotation matrix from a provided angle.

    Args:
        angle: The angle in radians.

    Returns:
        The rotation matrix
    """
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

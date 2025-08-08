import math
import numpy as np

from video_preprocessing.utils import (
    rotate_coords,
    xy_spline_smoothing,
    calculate_distance_history,
    distance_between_tuples,
)


def test_rotate_coords_90_degrees_origin():
    x = np.array([1.0, 0.0])
    y = np.array([0.0, 1.0])
    theta = math.pi / 2
    xr, yr = rotate_coords(x, y, theta, 0.0, 0.0)
    assert np.allclose(xr, np.array([0.0, -1.0]), atol=1e-7)
    assert np.allclose(yr, np.array([1.0, 0.0]), atol=1e-7)


def test_xy_spline_smoothing_shape_and_endpoints():
    points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    smoothed = xy_spline_smoothing(points, number_of_points=10, smoothing_factor=1)
    assert smoothed.shape == (10, 2)
    assert np.allclose(smoothed[0], points[0], atol=1e-6)
    assert np.allclose(smoothed[-1], points[-1], atol=1e-6)


def test_calculate_distance_history_simple():
    mass_centers = [(0, 0), (3, 4), (6, 8)]
    result = calculate_distance_history(mass_centers)
    assert np.allclose(result, np.array([5.0, 10.0]))


def test_distance_between_tuples_simple():
    assert distance_between_tuples((0, 0), (3, 4)) == 5.0



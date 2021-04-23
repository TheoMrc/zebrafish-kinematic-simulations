import numpy as np
from scipy.interpolate import splprep, splev
from itertools import groupby


def rotate_coords(x, y, theta, ox, oy):
    """Rotate arrays of coordinates x and y by theta radians about the
    point (ox, oy)."""
    sin, cos = np.sin(theta), np.cos(theta)
    x, y = np.asarray(x) - ox, np.asarray(y) - oy
    return x * cos - y * sin + ox, x * sin + y * cos + oy


def xy_spline_smoothing(data_array: np.ndarray, number_of_points: int, smoothing_factor: int = 2, ):
    data_array = np.array([i[0] for i in groupby([tuple(point) for point in data_array])])  # remove adjacent duplicates
    weights = np.ones(len(data_array.T[0]))
    weights[0] = 10
    weights[-1] = 10
    # noinspection PyTupleAssignmentBalance
    tck, u = splprep([data_array.T[0], data_array.T[1]], w=weights, k=3, s=smoothing_factor)
    new_points_repartition = np.linspace(0, 1, number_of_points)

    smoothed_points = splev(new_points_repartition, tck)
    return np.array(smoothed_points).T

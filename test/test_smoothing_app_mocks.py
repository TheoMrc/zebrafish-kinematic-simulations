import os
import matplotlib
import numpy as np

# Use a non-interactive backend for matplotlib
matplotlib.use("Agg")

from video_preprocessing.smoothing_app import (
    smooth_data_to_plot,
    update_graph,
)


class FakeVideo:
    def __init__(self, angles):
        self.angles = angles
        self.smoothed_angles = []


class DummyAxis:
    def __init__(self):
        self.cleared = False
        self.plots = []
        self.title = None

    def clear(self):
        self.cleared = True

    def plot(self, x, y, **kwargs):
        self.plots.append((np.asarray(x), np.asarray(y), kwargs))

    def legend(self):
        return None

    def set_title(self, title):
        self.title = title


class DummyCanvas:
    def __init__(self):
        self.drawn = False

    def draw(self):
        self.drawn = True


def test_smooth_data_to_plot_basic():
    # Create a simple monotonic angle series
    angles = np.linspace(0.0, 10.0, 50).tolist()
    video = FakeVideo(angles)

    # No exclusions
    xs, raw_cumul, smoothed_cumul, raw_delta, smoothed_delta = smooth_data_to_plot(
        video, excluded_data="", gaussian_sigma=0.5, spl_smoothing_factor=10.0
    )

    n = len(angles) - 1
    assert len(list(xs)) == n
    assert len(raw_cumul) == n
    assert len(smoothed_cumul) == n
    assert len(raw_delta) == n
    assert len(smoothed_delta) == n

    # Exclusion range shouldn't error and should introduce NaNs during smoothing phase
    xs2, *_ = smooth_data_to_plot(
        video, excluded_data="5, 10", gaussian_sigma=1.0, spl_smoothing_factor=5.0
    )
    assert len(list(xs2)) == n


def test_update_graph_calls_and_canvas_draw():
    angles = np.linspace(0.0, 5.0, 30).tolist()
    video = FakeVideo(angles)

    ax = [DummyAxis(), DummyAxis()]
    canvas = DummyCanvas()

    update_graph(
        video=video,
        excluded_data="2, 6",
        gaussian_sigma=0.8,
        spl_smoothing_factor=3.0,
        ax=ax,
        canvas=canvas,
    )

    # Both subplots should have been cleared and plotted to
    assert ax[0].cleared and ax[1].cleared
    assert len(ax[0].plots) > 0 and len(ax[1].plots) > 0
    assert ax[0].title is not None and ax[1].title is not None
    assert canvas.drawn is True



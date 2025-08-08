import pytest
import os
import pathlib
import matplotlib
import tkinter as tk
from video_preprocessing.smoothing_app import SmoothingApp, StartPage, smooth_data_to_plot
from video_preprocessing.experiment import Video
import numpy as np
import json


@pytest.fixture
def test_dir_path() -> str:
    yield pathlib.Path(__file__).parent.absolute()


@pytest.fixture(name="angles")
def load_test_angles(test_dir_path):
    with open(
        os.path.join(
            test_dir_path, "test_experiment", "test_data", "angles_test_data.json"
        ),
        "r",
    ) as data_file:
        angles = json.load(data_file)
        yield angles


@pytest.fixture(name="video")
def create_video_object(test_dir_path, angles):
    video_path = os.path.join(test_dir_path, "test_experiment", "test_video")
    video = Video(video_path, "tif")
    video.angles = angles
    yield video


def test_app_validates_instantly(video):
    # Use headless backend
    matplotlib.use("Agg")

    # Create the app window but prevent it from showing and blockless-close it
    app = SmoothingApp(video)
    app.withdraw()  # do not show a real window in CI

    # Compute smoothing immediately so the app does not rely on user input
    xs, raw_cumul, smoothed_cumul, raw_delta, smoothed_delta = smooth_data_to_plot(
        video, excluded_data="", gaussian_sigma=0.5, spl_smoothing_factor=10.0
    )
    assert len(list(xs)) == len(video.angles) - 1
    assert len(smoothed_delta) == len(video.angles) - 1

    # Immediately destroy the window to avoid hanging mainloop
    app.kill_app()
    # After destruction, tkinter raises TclError on winfo calls
    import pytest
    with pytest.raises(tk.TclError):
        _ = app.winfo_exists()

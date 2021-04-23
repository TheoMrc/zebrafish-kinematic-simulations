import pytest
import os
import pathlib
import matplotlib
from video_preprocessing.smoothing_app import SmoothingApp, StartPage
from video_preprocessing.experiment import Video
import numpy as np
import json


@pytest.fixture
def test_dir_path() -> str:
    yield pathlib.Path(__file__).parent.absolute()


@pytest.fixture(name='angles')
def load_test_angles(test_dir_path):
    with open(os.path.join(test_dir_path, 'test_experiment',
                           'test_data', 'angles_test_data.json'), 'r') as data_file:
        angles = json.load(data_file)
        yield angles


@pytest.fixture(name='video')
def create_video_object(test_dir_path, angles):
    video_path = os.path.join(test_dir_path, 'test_experiment', 'test_video')
    video = Video(video_path, 'tif')
    video.angles = angles
    yield video


def test_app(video):
    matplotlib.use("TkAgg")
    app = SmoothingApp(video)

    for widget in app.frames[StartPage].winfo_children():
        if widget['text'] == 'Update':
            app.bind('<Return>', lambda e: widget.invoke())
            break

    app.mainloop()
    assert video.smoothed_angles.any()
    print(video.angles)
    print(video.smoothed_angles)

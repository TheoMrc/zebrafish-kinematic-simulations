import os
import pathlib

import matplotlib
import numpy as np
import pytest

from video_preprocessing.experiment import Video, Frame


@pytest.fixture
def test_dir_path() -> str:
    return str(pathlib.Path(__file__).parent.absolute())


def test_process_frames_on_two_images(test_dir_path):
    video_path = os.path.join(test_dir_path, "test_experiment", "test_video")
    video = Video(video_path, img_extension="tif")
    # Load only first two frames for speed
    Video.read_frames(video, start_frame=0, end_frame=2, head_up=True, rot90=0)

    angles, mass_centers = Video.process_frames(video.frames, final_shift=(-20, 0))
    assert isinstance(angles, list) and isinstance(mass_centers, list)
    assert len(angles) == len(video.frames)
    assert len(mass_centers) == len(video.frames)


def test_rotate_and_center_frame_standalone(test_dir_path):
    # Use a single frame and fabricate minimal attributes
    img_path = os.path.join(
        test_dir_path, "test_experiment", "test_video", "film18_10000_40cm000001.tif"
    )
    f = Frame(0, img_path, head_up=True)
    # Minimal state for rotate_and_center_frame
    f.fish_zone = {(150, 150), (151, 150), (150, 151)}
    f.mass_center = (150, 150)

    rotated, zone = f.rotate_and_center_frame(0.0, shift=(-20, 0))
    assert isinstance(rotated, np.ndarray)
    assert rotated.shape == (200, 200)
    assert isinstance(zone, np.ndarray)


def test_plotting_methods_do_not_block():
    # Run with headless backend
    matplotlib.use("Agg")

    # Build a minimal frame with required arrays
    f = object.__new__(Frame)
    f.frame_n = 1
    f.raw_frame = np.zeros((10, 10))
    f.frame = np.ones((10, 10))
    f.boolean_frame = np.zeros((10, 10))
    f.centered_bool_frame = np.zeros((10, 10))
    f.rotated_bool_frame = np.zeros((10, 10))
    f.fish_zone = {(1, 1), (2, 2)}
    f.angle_to_vertical = 0.0

    import matplotlib.pyplot as plt

    # Monkeypatch plt.show to no-op
    old_show = plt.show
    try:
        plt.show = lambda *args, **kwargs: None
        f.plot_frame_processing()
        f.plot_final_frame()
    finally:
        plt.show = old_show



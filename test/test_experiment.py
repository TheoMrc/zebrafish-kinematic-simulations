import pytest
import os
import pathlib
from video_preprocessing.experiment import Video, Frame, get_zones, refine_fish_zone, add_to_fish_zone
import numpy as np


@pytest.fixture
def test_dir_path() -> str:
    yield pathlib.Path(__file__).parent.absolute()


def test_load_frames_paths(test_dir_path):
    print(pathlib.Path(__file__).parent.absolute())
    video_folder_path = os.path.join(test_dir_path, "test_experiment", "test_video")
    frames_paths = Video.load_frames_paths(video_folder_path, img_extension="tif")
    assert len(frames_paths) == 696


def test_init_frame(test_dir_path):
    first_frame_path = os.path.join(
        test_dir_path, "test_experiment", "test_video", "film18_10000_40cm000001.tif"
    )

    frame = Frame(1, first_frame_path, head_up=False)
    frames = [frame]


def test_init_video(test_dir_path):
    pass
    ...
    # Video.process_frames(frames)


def test_get_zones_simple():
    bool_array = np.zeros((4, 4), dtype=bool)
    bool_array[0, 0] = True
    bool_array[0, 1] = True
    bool_array[3, 3] = True
    zones = get_zones(bool_array)
    # We expect two disconnected zones
    assert len(zones) == 2
    assert { (0,0), (0,1) } in zones


def test_refine_and_add_to_fish_zone():
    frame = np.array(
        [
            [10, 12, 13, 15],
            [12, 11, 14, 60],
            [13, 14, 10, 70],
            [16, 18, 12,  9],
        ],
        dtype=int,
    )
    fish_zone = {(0, 0)}
    refined = refine_fish_zone(frame, set(fish_zone))
    # Should at least include the starting point
    assert (0, 0) in refined

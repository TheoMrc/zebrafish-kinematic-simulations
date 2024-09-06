import pytest
import os
import pathlib
from video_preprocessing.experiment import Video, Frame


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

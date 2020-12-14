import pytest
import os
import pathlib
from video_preprocessing.experiment import Video, Frame


@pytest.fixture
def test_dir_path() -> str:
    yield pathlib.Path(__file__).parent.absolute()


def test_load_frames_paths(test_dir_path):
    video_folder_path = os.path.join(test_dir_path, 'test_experiment', 'test_video', 'jpg_folder')
    frames_paths = Video.load_frames_paths(video_folder_path, img_extension='jpg')
    assert len(frames_paths) == 696


def test_init_and_plot_frame(test_dir_path):
    first_frame_path = os.path.join(test_dir_path, 'test_experiment', 'test_video',
                                    'jpg_folder', 'film18_10000_40cm000001.jpg')
    first_frame = Frame(1, first_frame_path, head_up=False)
    first_frame.plot_frame()

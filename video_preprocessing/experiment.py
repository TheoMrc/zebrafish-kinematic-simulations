from __future__ import annotations
import sys

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Set, Optional
from itertools import product
from scipy.ndimage.interpolation import rotate
from tqdm import tqdm

sys.setrecursionlimit(512*416)

# Coordinate = Tuple[int, int]


def get_zones(bool_array: np.array) -> List[Set[Tuple[int, int]]]:
    seen_pixels = set()
    zones = list()
    for x, y in product(*map(range, bool_array.shape)):
        if bool_array[x, y] and (x, y) not in seen_pixels:
            zone = set()
            add_to_zone(x, y, zone, bool_array, seen_pixels)
            zones.append(zone)
    return zones


def add_to_zone(x: int, y: int, zone: Set[Tuple[int, int]], bool_array: np.array,
                seen_pixels: Set[Tuple[int, int]]) -> None:
    seen_pixels.add((x, y))
    zone.add((x, y))
    for offset_x, offset_y in zip([1, -1, 0, 0], [0, 0, 1, -1]):
        if (0 <= x + offset_x < bool_array.shape[0] and 0 <= y + offset_y < bool_array.shape[1]
                and bool_array[x + offset_x, y + offset_y]
                and (x + offset_x, y + offset_y) not in seen_pixels):
            add_to_zone(x + offset_x, y + offset_y, zone, bool_array, seen_pixels)


class Video:
    def __init__(self, video_path: str, img_extension: str = 'tif'):
        self.path = video_path
        self.frames_paths = Video.load_frames_paths(video_path, img_extension)
        self.frames = list()
        self.angles = list()

    @classmethod
    def load_frames_paths(cls, video_path: str, img_extension: str = 'tif') -> List[str]:
        frames_paths = [os.path.join(video_path, file) for file in filter(lambda f: f.split('.')[-1] == img_extension,
                                                                          os.listdir(video_path))]
        print(f'\n{len(frames_paths)} frames in video folder')
        return frames_paths

    @classmethod
    def read_frames(cls, video, start_frame: Optional[int] = None,
                    end_frame: Optional[int] = None, head_up=True) -> None:
        for n_frame, path in enumerate(video.frames_paths[start_frame:end_frame]):
            frame = Frame(n_frame, path, head_up=head_up)
            video.frames.append(frame)
        print(f'\n{len(video.frames)} frames loaded')

    @classmethod
    def process_frames(cls, frames: List[Frame]):
        for frame in tqdm(frames, total=len(frames), desc="Image processing "):
            frame.boolean_frame = Frame.threshold(frame.frame, thresh=200)
            frame.fish_zone = Frame.get_fish_zone(frame.boolean_frame)
            frame.centered_bool_frame, frame.mass_center = Frame.create_fish_centered_frame(frame.fish_zone,
                                                                                            half_size=150)
            frame.angle_to_vertical = frame.calculate_rotation_angle()
            frame.rotated_bool_frame, frame.fish_zone = frame.rotate_and_center_frame()
            frame.plot_frame_processing()
            frame.plot_final_frame()


class Frame:
    def __init__(self, frame_n: int, frame_path: str, head_up: True):
        self.frame_n = frame_n
        self.path = frame_path
        self.raw_frame = plt.imread(frame_path)
        if not head_up:
            self.raw_frame = np.flipud(self.raw_frame)
        # self.frame = Frame.rgb2gray(Frame.hist_adjust(self.raw_frame))
        self.frame = Frame.rgb2gray(Frame.hist_adjust(Frame.image_standardisation(self.raw_frame)))

        self.shape = self.frame.shape
        self.boolean_frame = np.array([])
        self.centered_bool_frame = np.array([])
        self.rotated_bool_frame = np.array([])
        self.fish_zone = list()
        self.mass_center = tuple()
        self.angle_to_vertical = float()

    @classmethod
    def hist_adjust(cls, frame: np.array, gamma: float = 1):
        x, a, b, c, d = frame, frame.min(), frame.max(), 0, 255
        y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
        return y

    @classmethod
    def image_standardisation(cls, frame):
        return frame-np.array([np.mean(frame)])/np.array([np.std(frame)])


    @classmethod
    def rgb2gray(cls, frame: np.array):
        return np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])

    @classmethod
    def threshold(cls, frame: np.array, thresh: int = 200):
        return frame < thresh

    @classmethod
    def get_fish_zone(cls, boolean_frame: np.array) -> List[Tuple]:
        zones = get_zones(boolean_frame)
        fish_zone = list(max(zones, key=len))
        return fish_zone

    @classmethod
    def create_fish_centered_frame(cls, fish_zone: List[Tuple], half_size: int = 150) -> Tuple[np.array, Tuple]:
        x_list = [x for x, y in fish_zone]
        y_list = [y for x, y in fish_zone]
        mass_center = (np.mean(x_list).round(), np.mean(y_list).round())
        centered_bool_frame = np.zeros([2 * half_size, 2 * half_size])
        zone_arr = np.array(fish_zone) - np.array(mass_center).astype(int) + np.array((half_size, half_size))
        centered_bool_frame[tuple(zone_arr.T)] = 1
        return centered_bool_frame, mass_center

    def calculate_rotation_angle(self) -> float:
        norm_fish_zone = np.array(self.fish_zone) - np.array(self.mass_center).astype(int)
        # for every n point in fish, calculate angle between the (n, mass_center) straight line and the vertical line
        angles = np.arctan2(norm_fish_zone.T[0], norm_fish_zone.T[1])

        dist_list = list()
        for pixel in norm_fish_zone:
            dist_list.append(np.linalg.norm(pixel))
        dist_arr = np.array(dist_list)
        weighted_angles = dist_arr * angles / dist_arr.sum() * 180 / np.pi

        return sum(weighted_angles)

    def rotate_and_center_frame(self) -> Tuple[np.array, List[Tuple]]:
        # rotated_frame = rotate(self.centered_bool_frame, np.sum(self.angle_to_vertical),
        #                        reshape=False).round().astype(int)

        rotated_frame = rotate(self.centered_bool_frame, np.sum(self.angle_to_vertical),
                               reshape=False) > 0.05
        fish_zone = Frame.get_fish_zone(rotated_frame)
        rotated_frame, _ = Frame.create_fish_centered_frame(fish_zone, half_size=150)
        return rotated_frame, fish_zone

    def plot_frame_processing(self):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 7))
        ax = axes.ravel()
        plt.gray()

        ax[0].imshow(self.raw_frame, cmap="gray")
        ax[0].set_title(f'Raw frame {self.frame_n}')
        ax[1].imshow(self.frame, cmap="gray")
        ax[1].set_title(f'Grayscale and adjusted frame {self.frame_n}')
        ax[2].imshow(self.boolean_frame, cmap="gray")
        ax[2].set_title(f'Threshold frame {self.frame_n}')
        ax[3].imshow(self.centered_bool_frame, cmap="gray")
        ax[3].set_title(f'Centered frame {self.frame_n}')

        for a in ax:
            a.axis('off')
        plt.show()

    def plot_final_frame(self):
        plt.figure(figsize=(6, 6))
        plt.imshow(self.rotated_bool_frame)
        plt.grid()
        plt.title(f'Rotated frame {self.frame_n}')
        plt.show()

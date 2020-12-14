from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Set, Dict, Union
from itertools import product


Coordinate = Tuple[int, int]


def get_zones(bool_array: np.array) -> List[Set[Tuple[int, int]]]:
    seen_pixels = set()
    zones = list()
    for x, y in product(*map(range, bool_array.shape)):
        if bool_array[x, y] and (x, y) not in seen_pixels:
            zone = set()
            add_to_zone(x, y, zone, bool_array, seen_pixels)
            zones.append(zone)
    return zones


def add_to_zone(x: int, y: int, zone: Set[Coordinate], bool_array: np.array, seen_pixels: Set[Coordinate]) -> None:
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
        self.frames = Video.load_frames_paths(video_path, img_extension)

    @classmethod
    def load_frames_paths(cls, video_path: str, img_extension: str = 'tif') -> List[np.array]:
        frames_paths = [os.path.join(video_path, file) for file in filter(lambda f: f.split('.')[-1] == img_extension,
                                                                          os.listdir(video_path))]
        print(f'{len(frames_paths)} loaded')
        return frames_paths


class Frame:
    def __init__(self, frame_n: int, frame_path: str):
        self.frame_n = frame_n
        self.path = frame_path
        self.raw_frame = plt.imread(frame_path)
        self.frame = Frame.rgb2gray(Frame.hist_adjust(self.raw_frame))
        self.shape = self.frame.shape
        self.boolean_frame = Frame.threshold(self.frame, thresh=190)
        self.fish_zone = Frame.get_fish_zone(self.boolean_frame)
        self.centered_bool_frame = Frame.create_fish_centered_frame(self.fish_zone, half_size=150)

    @classmethod
    def hist_adjust(cls, frame: np.array, gamma=1):
        x, a, b, c, d = frame, frame.min(), frame.max(), 0, 1
        y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
        return y

    @classmethod
    def rgb2gray(cls, frame: np.array):
        return np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])

    @classmethod
    def threshold(cls, frame: np.array, thresh: int = 190):
        return frame < thresh / 255

    @classmethod
    def get_fish_zone(cls, boolean_frame: np.array) -> List[Tuple]:
        zones = get_zones(boolean_frame)
        fish_zone = list(max(zones, key=len))
        return fish_zone

    @classmethod
    def create_fish_centered_frame(cls, fish_zone, half_size: int = 150):
        x_list = [x for x, y in fish_zone]
        y_list = [y for x, y in fish_zone]
        mass_center = (np.mean(x_list), np.mean(y_list))
        bool_frame_center = np.zeros([2 * half_size, 2 * half_size])
        zone_arr = np.array(fish_zone) - np.array(mass_center).astype(int) + np.array((half_size, half_size))
        bool_frame_center[tuple(zone_arr.T)] = 1
        return bool_frame_center

    def plot_frame(self):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 7))
        ax = axes.ravel()
        plt.gray()

        ax[0].imshow(self.raw_frame, cmap="gray")
        ax[0].set_title('Raw frame')
        ax[1].imshow(self.frame, cmap="gray")
        ax[1].set_title('Grayscale and adjusted frame')
        ax[2].imshow(self.boolean_frame, cmap="gray")
        ax[2].set_title('Threshold frame')
        ax[3].imshow(self.centered_bool_frame, cmap="gray")
        ax[3].set_title('Centered frame')

        for a in ax:
            a.axis('off')
        plt.show()





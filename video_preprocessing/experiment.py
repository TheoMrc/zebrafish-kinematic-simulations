from __future__ import annotations
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Set, Optional
from itertools import product
from scipy.signal import convolve2d
from tqdm import tqdm
from video_preprocessing.utils import rotate_coords
import cv2
import math
sys.setrecursionlimit(512*416*2)


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


# noinspection PyTypeChecker
class Video:
    def __init__(self, video_path: str, img_extension: str = 'tif'):
        self.path = video_path
        self.frames_paths = Video.load_frames_paths(video_path, img_extension)

        self.frames = list()
        self.angles = list()
        self.smoothed_angles = list()

    @classmethod
    def load_frames_paths(cls, video_path: str, img_extension: str = 'tif') -> List[str]:
        frames_paths = [os.path.join(video_path, file) for file in filter(lambda f: f.split('.')[-1] == img_extension,
                                                                          os.listdir(video_path))]
        print(f'\n{len(frames_paths)} frames in video folder')
        return frames_paths

    @classmethod
    def read_frames(cls, video: Video, start_frame: Optional[int] = None,
                    end_frame: Optional[int] = None, head_up=True) -> None:
        for n_frame, path in tqdm(enumerate(video.frames_paths[start_frame:end_frame]),
                                  total=len(video.frames_paths[start_frame:end_frame]),
                                  desc='Reading frames'):
            frame = Frame(n_frame, path, head_up=head_up)
            video.frames.append(frame)
        print(f'\n{len(video.frames)} frames loaded')

    @classmethod
    def process_frames(cls, frames: List[Frame]) -> List[float]:
        angles = list()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        for frame in tqdm(frames, total=len(frames), desc="Image processing "):

            frame.boolean_frame = frame.frame <= np.quantile(frame.frame, .01)

            frame.fish_zone = np.array(list(Frame.get_fish_zone(frame.boolean_frame)))

            raw_fish_image = np.zeros([416, 512])
            raw_fish_image[tuple(frame.fish_zone.T)] = 1
            closed_fish_image = cv2.morphologyEx(raw_fish_image, cv2.MORPH_CLOSE, kernel)

            frame.fish_zone = np.argwhere(closed_fish_image == 1)

            frame.centered_bool_frame, frame.mass_center = Frame.create_fish_centered_frame(frame.fish_zone,
                                                                                            half_size=150)
            frame.raw_centered_bool_frame = frame.centered_bool_frame.copy()
            frame.fish_zone = np.argwhere(frame.centered_bool_frame == 1)
            frame.mass_center = (150, 150)

            if frame.frame_n > 0:
                frame.centered_bool_frame = cv2.morphologyEx(frame.centered_bool_frame, cv2.MORPH_CLOSE, kernel)
                frame.centered_bool_frame, frame.fish_zone = frame.rotate_and_center_frame(
                    frames[frame.frame_n - 1].angle_to_vertical)

                frame.centered_bool_frame = cv2.morphologyEx(frame.centered_bool_frame, cv2.MORPH_CLOSE, kernel)
                frame.fish_zone = np.argwhere(frame.centered_bool_frame == 1)

            frame.angle_to_vertical = frame.calculate_rotation_angle(frame.fish_zone,
                                                                     (np.mean(frame.fish_zone.T[0]),
                                                                      np.mean(frame.fish_zone.T[1])))

            frame.rotated_bool_frame, frame.fish_zone = frame.rotate_and_center_frame(frame.angle_to_vertical)

            if frame.frame_n > 0:
                frame.rotated_bool_frame = cv2.morphologyEx(frame.rotated_bool_frame, cv2.MORPH_CLOSE, kernel)

            for n in range(2):
                counts = convolve2d(frame.rotated_bool_frame, np.ones((3, 3)), mode='same')
                frame.rotated_bool_frame[counts >= 4] = 1

            if frame.frame_n > 0:
                frame.angle_to_vertical += frames[frame.frame_n - 1].angle_to_vertical
            frame.rotated_bool_frame = cv2.morphologyEx(frame.rotated_bool_frame, cv2.MORPH_CLOSE, kernel)

            angles.append(frame.angle_to_vertical)
        return angles

    @classmethod
    def process_frames_from_smoothed_angle(cls, frames: List[Frame], smoothed_angles: List[float],
                                           target_path: Optional[str], save_frames: bool = True):

        for frame, angle in tqdm(zip(frames, smoothed_angles), total=len(frames),
                                 desc="Rotating images with smoothed angles"):
            frame.fish_zone = np.argwhere(frame.raw_centered_bool_frame != 0)
            frame.mass_center = (150, 150)
            frame.rotated_bool_frame, frame.fish_zone = frame.rotate_and_center_frame(angle)
            for n in range(2):
                counts = convolve2d(frame.rotated_bool_frame, np.ones((3, 3)), mode='same')
                frame.rotated_bool_frame[counts >= 4] = 1

            if save_frames:
                np.savetxt(os.path.join(target_path, f'frame_{frame.frame_n:05}.dat'),
                           frame.rotated_bool_frame, delimiter=',', fmt="%5i")


class Frame:
    def __init__(self, frame_n: int, frame_path: str, head_up: True):
        self.frame_n = frame_n
        self.path = frame_path
        if frame_path.split('.')[-1] == 'dat':
            self.raw_frame = [px for px in open(frame_path)]
            self.raw_frame = np.array(self.raw_frame).reshape((300, 300)).astype(int)
        else:
            self.raw_frame = plt.imread(frame_path)
        if not head_up:
            self.raw_frame = np.flipud(self.raw_frame)
        self.frame = self.raw_frame

        self.previous_angle = None
        self.clipped_frame = np.array([])
        self.boolean_frame = np.array([])
        self.raw_centered_bool_frame = np.array([])

        self.centered_bool_frame = np.array([])
        self.rotated_bool_frame = np.array([])
        self.fish_zone = set()
        self.mass_center = tuple()
        self.angle_to_vertical = float()

    @classmethod
    def hist_adjust(cls, frame: np.array, gamma: float = 1):
        x, a, b, c, d = frame, frame.min(), frame.max(), 0, 255

        y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
        return y.astype(float)

    @classmethod
    def image_standardisation(cls, frame):
        return frame - np.array([np.mean(frame)]) / np.array([np.std(frame)])

    @classmethod
    def rgb2gray(cls, frame: np.array):
        return np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])

    @classmethod
    def get_fish_zone(cls, boolean_frame: np.array) -> Set[Tuple[int, int]]:
        zones = get_zones(boolean_frame)
        fish_zone = set(max(zones, key=len))
        return fish_zone

    @classmethod
    def create_fish_centered_frame(cls, fish_zone: np.ndarray,
                                   half_size: int = 150, shift: Tuple = (0, 0)) -> Tuple[np.array, Tuple]:

        mass_center = np.mean(fish_zone.T[0]).round(), np.mean(fish_zone.T[1]).round()
        centered_bool_frame = np.zeros([2 * half_size, 2 * half_size])
        zone_arr = fish_zone - np.round(np.array(mass_center)).astype(int) + np.array((half_size, half_size)) + shift
        centered_bool_frame[tuple(zone_arr.T)] = 1

        return centered_bool_frame, mass_center

    @classmethod
    def calculate_rotation_angle(cls, fish_zone, mass_center) -> float:
        norm_fish_zone = np.array(list(fish_zone)) - np.round(np.array(mass_center))

        angles = list()
        dist_list = list()
        matlab_norm_fish_zone = norm_fish_zone * (-1, 1)
        for pixel in matlab_norm_fish_zone:
            angles.append(math.atan(pixel[1]/(pixel[0] + 1E-12)))
            dist_list.append(np.linalg.norm(pixel))
        dist_arr = np.array(dist_list)

        weighted_angles = dist_arr * np.round(np.array(angles), 6)

        return np.sum(weighted_angles) / dist_arr.sum() * 180 / np.pi

    def rotate_and_center_frame(self, degrees_angle) -> Tuple[np.array, Set[Tuple[int, int]]]:

        normalized_zone = np.array(list(self.fish_zone)) - np.round(np.array(self.mass_center)).astype(int)
        rotated_zone = np.round(np.array(rotate_coords(normalized_zone.T[0], normalized_zone.T[1],
                                                       degrees_angle * np.pi / 180,
                                                       0, 0))).T.astype(int) + np.round(np.array(self.mass_center)).astype(int)

        rotated_frame, _ = Frame.create_fish_centered_frame(rotated_zone, half_size=100, shift=(-20, 0))

        return rotated_frame, rotated_zone

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
        plt.figure('final frame', figsize=(6, 6))
        plt.imshow(self.rotated_bool_frame, cmap='viridis')
        plt.contour(self.rotated_bool_frame, linewidths=1, cmap='viridis')
        plt.grid()
        plt.title(f'Rotated frame {self.frame_n} \n surf_before = {len(self.fish_zone)} '
                  f'; surf_after = {np.sum(self.rotated_bool_frame)}\n, angle = {self.angle_to_vertical}')
        plt.show()


def refine_fish_zone(frame: np.array, fish_zone: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    for x, y in list(fish_zone):
        add_to_fish_zone(x, y, fish_zone, frame)
    return fish_zone


def add_to_fish_zone(x: int, y: int, fish_zone: Set[Tuple[int, int]], frame: np.array) -> None:
    offsets = zip([1, -1, 0, 0, 1, 1, -1, -1], [0, 0, 1, -1, 1, -1, 1, -1])

    for offset_x, offset_y in offsets:
        if (x + offset_x, y + offset_y) in fish_zone:
            pass
        elif (0 <= x + offset_x < frame.shape[0] and 0 <= y + offset_y < frame.shape[1]
              and abs(frame[x, y] - frame[x + offset_x, y + offset_y]) < 40
              and frame[x + offset_x, y + offset_y] < np.median(frame) - 12
              and (x + offset_x, y + offset_y) not in fish_zone):
            fish_zone.add((x + offset_x, y + offset_y))
            add_to_fish_zone(x + offset_x, y + offset_y, fish_zone, frame)


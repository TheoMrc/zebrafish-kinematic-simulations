import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from video_preprocessing.experiment import Video


if os.environ['COMPUTERNAME'] == 'DESKTOP-H65TDGH':
    video_folder_path = r'E:\Documents\OneDrive\Documents\Python_projects\MRGM\Guillaume\film18_10000_40cm'
else:
    video_folder_path = r'D:\Users\Théo\Documents\OneDrive\Documents\Python_projects\MRGM\Guillaume\film18_10000_40cm'

if os.environ['COMPUTERNAME'] == 'DESKTOP-H65TDGH':
    video_folder_path = r'E:\Documents\OneDrive\Documents\Python_projects\MRGM\Guillaume\test_data\centered'
else:
    video_folder_path = r'D:\Users\Théo\Documents\OneDrive\Documents\Python_projects\MRGM\Guillaume\test_data\centered'

video = Video(video_folder_path, img_extension='dat')
Video.read_frames(video, start_frame=0, end_frame=1, head_up=False)
video.angles = Video.process_frames(video.frames)
# arr = np.array([np.array(range(len(video.angles))), np.array(video.angles)]).T
# # smoothed = xy_spline_smoothing(arr, 565, 10)
smoothed = gaussian_filter1d(np.array(video.angles), sigma=4)
plt.plot(range(len(video.angles)), video.angles, label=f'Raw angles {np.std(video.angles)}')
plt.plot(range(len(smoothed)), smoothed, label='Smoothed angles')
plt.legend()
plt.show()

delta_angles = np.array(video.angles)[1:] - np.array(video.angles)[:-1]
smoothed_delta_angles = gaussian_filter1d(delta_angles, sigma=1)
delta_smoothed_angles = np.array(smoothed)[1:] - np.array(smoothed)[:-1]
plt.plot(range(len(delta_angles)), delta_angles, label=f'Delta angles {np.std(delta_angles)}')
# plt.plot(range(len(smoothed_delta_angles)), smoothed_delta_angles, label='Smoothed Delta angles')
# plt.plot(range(len(delta_smoothed_angles)), delta_smoothed_angles, label='Delta Smoothed angles')

plt.legend()
plt.show()

surfaces = np.array([len(frame.fish_zone) for frame in video.frames])
plt.plot(range(len(surfaces)), surfaces, label='surfaces')
plt.legend()

plt.show()

guillaume_angles = [alpha for alpha in open(r"D:\Users\Théo\Documents\OneDrive\Documents\Python_projects\MRGM\Guillaume\test_data\Angles_000001.dat")]
frame = video.frames[0]

fish_zone = np.argwhere(frame.centered_bool_frame == 1)
mass_center = (np.mean(fish_zone.T[0]), np.mean(fish_zone.T[1]))

norm_fish_zone = np.array(list(fish_zone)) - np.round(np.array(mass_center)).astype(int)
# for every n point in fish, calculate angle between the (n, mass_center) straight line and the vertical line
angles = np.arctan(- norm_fish_zone.T[1] / (norm_fish_zone.T[0] + 1E-12))
# angles = list()
dist_list = list()
for pixel in norm_fish_zone:
    # angles.append(- math.atan(pixel[1]/(pixel[0] + 1E-12)))
    dist_list.append(np.linalg.norm(pixel))
dist_arr = np.array(dist_list)

print('\n', dist_arr.sum())
weighted_angles = dist_arr * angles
print(weighted_angles.sum())

print(np.sum(weighted_angles) / dist_arr.sum() * 180 / np.pi)
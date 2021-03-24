import os
import matplotlib.pyplot as plt
import numpy as np
from video_preprocessing.experiment import Video
from video_preprocessing.utils import xy_spline_smoothing

if os.environ['COMPUTERNAME'] == 'DESKTOP-H65TDGH':
    video_folder_path = r'E:\Documents\OneDrive\Documents\Python_projects\MRGM\Guillaume\film18_10000_40cm'
else:
    video_folder_path = r'D:\Users\Théo\Documents\OneDrive\Documents\Python_projects\MRGM\Guillaume\film18_10000_40cm'
video = Video(video_folder_path, img_extension='tif')
Video.read_frames(video, start_frame=0, end_frame=565, head_up=False)
video.angles = Video.process_frames(video.frames)
delta_angles = np.array(video.angles)[1:] - np.array(video.angles)[:-1]
arr = np.array([np.array(range(len(video.angles))), np.array(video.angles)]).T
smoothed = xy_spline_smoothing(arr, 565, 10)
plt.plot(range(len(video.angles)), video.angles, label='Raw angles')
plt.plot(smoothed.T[0], smoothed.T[1], label='Smoothed angles')

plt.legend()
plt.show()



plt.plot(range(len(delta_angles)), delta_angles, label='Delta angles')

plt.legend()

plt.show()

surfaces = np.array([len(frame.fish_zone) for frame in video.frames])
plt.plot(range(len(surfaces)), surfaces, label='surfaces')
plt.legend()

plt.show()
print(len(surfaces))
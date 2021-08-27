import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from video_preprocessing.experiment import Video
from video_preprocessing.smoothing_app import SmoothingApp, StartPage
from tkinter import ttk
import matplotlib

print(matplotlib.get_backend())
if os.environ['COMPUTERNAME'] == 'DESKTOP-H65TDGH':
    video_folder_path = r'E:\Documents\OneDrive\Documents\Python_projects\MRGM\Guillaume\film18_10000_40cm'
    target_path = r'E:\Documents\OneDrive\Documents\Python_projects\MRGM\Guillaume\film18_10000_40cm\pyth_results'
    os.makedirs(target_path, exist_ok=True)
    print(video_folder_path, '\n', target_path)


elif os.environ['COMPUTERNAME'] == 'PC-STAGIARES-2':
    video_folder_path = r"X:\SAUVEGARDE\Théo\2021_08_23 DMSO DZP Dex\DZP_50_µM_4"
    target_path = os.path.join(r"V:\Sauvegarde\Theo\Modelisation data", video_folder_path.split(os.path.sep)[-1])

    # video_folder_path = r'X:\SAUVEGARDE\Théo\2020_12_09 DMSO CPO film simu 3-9% Dex\CPO_150_nM_3percent_fish_1_1'
    # target_path = r'V:\Sauvegarde\Theo\Modelisation data\CPO_150_nM_3percent_fish_1_1'
    os.makedirs(target_path, exist_ok=True)

    print(video_folder_path, '\n', target_path)

else:
    sys.exit()

video = Video(video_folder_path, img_extension='tif')
Video.read_frames(video, start_frame=200, end_frame=1000, head_up=True)

video.angles = Video.process_frames(video.frames)


surfaces = np.array([len(frame.fish_zone) for frame in video.frames])
plt.plot(range(len(surfaces)), surfaces, label='surfaces')
plt.legend()
plt.show()

matplotlib.use("TkAgg")
app = SmoothingApp(video)
for widget in app.frames[StartPage].winfo_children():
    if widget['text'] == 'Update':
        app.bind('<Return>', lambda e: widget.invoke())
        break

app.mainloop()

Video.process_frames_from_smoothed_angle(video.frames, video.smoothed_angles, target_path=target_path, save_frames=True)

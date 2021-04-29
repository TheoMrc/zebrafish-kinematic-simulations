import os
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

else:
    video_folder_path = r'D:\Users\Théo\Documents\OneDrive\Documents\Python_projects\MRGM\Guillaume\film18_10000_40cm'
    target_path = r'D:\Users\Théo\Documents\OneDrive\Documents\Python_projects\MRGM\Guillaume\film18_10000_40cm\pyth_results'
    os.makedirs(target_path, exist_ok=True)

video = Video(video_folder_path, img_extension='tif')
Video.read_frames(video, start_frame=0, end_frame=565, head_up=False)
video.angles = Video.process_frames(video.frames)


surfaces = np.array([len(frame.fish_zone) for frame in video.frames])
plt.plot(range(len(surfaces)), surfaces, label='surfaces')
plt.legend()
plt.show()

matplotlib.use("TkAgg")
app = SmoothingApp(video)
for widget in app.frames[StartPage].winfo_children():
    if isinstance(widget, ttk.Button):
        app.bind('<Return>', lambda e: widget.invoke())
        break

app.mainloop()

Video.process_frames_from_smoothed_angle(video.frames, video.smoothed_angles, target_path=target_path, save_frames=True)

import os
from video_preprocessing.experiment import Video

if os.environ['COMPUTERNAME'] == 'DESKTOP-H65TDGH':
    video_folder_path = r'E:\Documents\OneDrive\Documents\Python_projects\MRGM\Guillaume\film18_10000_40cm\film18_10000_40cm'
else:
    video_folder_path = r'D:\Users\Théo\Documents\OneDrive\Documents\Python_projects\MRGM\Guillaume\film18_10000_40cm\film18_10000_40cm'
video = Video(video_folder_path, img_extension='jpg')
Video.read_frames(video, start_frame=0, end_frame=10, head_up=False)
Video.process_frames(video.frames)

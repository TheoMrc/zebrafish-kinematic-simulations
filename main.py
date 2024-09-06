import os
import matplotlib.pyplot as plt
import numpy as np
from video_preprocessing.experiment import Video
from video_preprocessing.smoothing_app import SmoothingApp, StartPage
from video_preprocessing.utils import calculate_distance_history
from video_preprocessing.video_config import video_config_dict
import matplotlib

show_frames = True

print(matplotlib.get_backend(), os.environ["COMPUTERNAME"])
source_path = r"test\test_experiment"
target_main_folder = r"test\tmp_results"

for video_name in ["test_video"]:

    #######################################
    # Parameters :
    head_up = video_config_dict[video_name]["head_up"]
    rot90 = video_config_dict[video_name]["rot90"]
    shift = video_config_dict[video_name]["shift"]
    #######################################

    video_folder_path = os.path.join(source_path, video_name)
    # target_path = os.path.join(r"V:\Sauvegarde\Theo\Modelisation data", video_name)
    target_path = os.path.join(target_main_folder, video_name)
    os.makedirs(os.path.join(target_path, "kinematics_data"), exist_ok=True)
    print("\n\n", video_folder_path, "\n", target_path)

    video = Video(video_folder_path, img_extension="tif")
    Video.read_frames(
        video, start_frame=0, end_frame=1000, head_up=head_up, rot90=rot90
    )

    video.angles, video.mass_centers = Video.process_frames(
        video.frames, final_shift=shift
    )

    distances_history = np.array(calculate_distance_history(video.mass_centers))
    surfaces_history = np.array([len(frame.fish_zone) for frame in video.frames])

    for label, history in zip(
        ["Surface", "Distance"], [surfaces_history, distances_history]
    ):

        plt.plot(range(len(history)), history, label=label)
        plt.legend()
        plt.show()
        np.savetxt(
            os.path.join(target_path, "kinematics_data", f"{label}.dat"), history
        )
        print(os.path.join("kinematics_data", f"{label}.dat"), "saved!")

    plt.plot(
        np.array(video.mass_centers).T[0],
        np.array(video.mass_centers).T[1],
        label="mass center",
    )
    plt.legend()
    plt.show()
    np.savetxt(
        os.path.join(target_path, "kinematics_data", f"mass_center_pos.dat"),
        video.mass_centers,
    )
    print(os.path.join("kinematics_data", f"mass_center_pos.dat"), "saved!")

    app = SmoothingApp(video)
    for widget in app.frames[StartPage].winfo_children():
        if widget["text"] == "Update":
            app.bind("<Return>", lambda e: widget.invoke())
            break

    app.mainloop()

    np.savetxt(
        os.path.join(target_path, "kinematics_data", f"Angle.dat"),
        np.array(video.smoothed_angles),
    )
    print(os.path.join("kinematics_data", f"Angle.dat"), "saved!")

    Video.process_frames_from_smoothed_angle(
        video.frames,
        video.smoothed_angles,
        final_shift=shift,
        target_path=target_path,
        save_frames=True,
    )
    print("Total distance =", distances_history[-1])
    if show_frames:
        for frame in video.frames:
            frame.plot_frame_processing()

    os.startfile(target_path)

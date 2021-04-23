import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

from video_preprocessing.experiment import Video


class SmoothingApp(tk.Tk):

    def __init__(self, video: Video, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.wm_title(self, "Smooth parameters tuner")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        frame = StartPage(container, self, video)

        self.frames[StartPage] = frame

        frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class PopUpConfirmQuit(tk.Toplevel):
    """A TopLevel popup that asks for confirmation that the user wants to quit.
    Upon confirmation, the App is destroyed.
    If not, the popup closes and no further action is taken
    """
    def __init__(self, root, video, smoothed_angles):
        super().__init__(root)
        self.title("Quit")
        self.geometry(f"300x90+810+490")
        l1 = ttk.Label(self, image="::tk::icons::question")
        l1.grid(row=0, column=0, pady=(7, 0), padx=(10, 30), sticky="e")
        l2 = ttk.Label(self, text="Validate smoothing parameters ?")
        l2.grid(row=0, column=1, columnspan=3, pady=(7, 10), sticky="w")

        b1 = ttk.Button(self, text="Yes", command=root.quit, width=10)
        b1.grid(row=1, column=1, padx=(2, 35), sticky="e")
        b2 = ttk.Button(self, text="No", command=self.destroy, width=10)
        b2.grid(row=1, column=2, padx=(2, 35), sticky="e")


class StartPage(tk.Frame):

    def __init__(self, parent, controller, video):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Angle graph smoothing", font=("Calibri", 12))
        label.pack(pady=10, padx=10)
        validate_button = ttk.Button(self, text="Validate Settings",
                                     command=lambda: PopUpConfirmQuit(controller, video, smoothed_cumul_angle))
        validate_button.pack(side=tk.BOTTOM, pady=5)
        E2 = ttk.Entry(self)
        E2.insert(0, str(31))
        E2.pack(side=tk.BOTTOM, pady=2)
        label2 = tk.Label(self, text="UnivariateSpline smoothing parameter")
        label2.pack(side=tk.BOTTOM, padx=10)

        E1 = ttk.Entry(self)
        E1.pack(side=tk.BOTTOM, pady=2)
        E1.insert(0, str(.5))
        label1 = tk.Label(self, text="Gaussian filtering sigma")
        label1.pack(side=tk.BOTTOM, padx=10)

        E3 = ttk.Entry(self)
        E3.insert(0, '200 , 250')
        E3.pack(side=tk.BOTTOM, pady=2)
        label1 = tk.Label(self, text="Coma separated exclusion zones limits")
        label1.pack(side=tk.BOTTOM, padx=10)

        button1 = ttk.Button(self, text="Update",
                             command=lambda: update_graph(video, E3.get(), E1.get(), E2.get(), ax, canvas))
        button1.pack(side=tk.BOTTOM)
        (all_xs, raw_cumul_angle, smoothed_cumul_angle, raw_delta_angles,
         smoothed_delta_angles) = smooth_data_to_plot(video, E3.get(), E1.get(), E2.get())

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
        ax = axes.ravel()

        ax[0].plot(all_xs, raw_cumul_angle, alpha=.7, label='Raw')
        ax[0].plot(all_xs, smoothed_cumul_angle, alpha=.7, label='Smoothed')
        ax[0].legend()
        ax[0].set_title('Cumulative rotation angle = f(t)')

        ax[1].plot(all_xs, raw_delta_angles, alpha=.7, label='Raw')
        ax[1].plot(all_xs, smoothed_delta_angles, alpha=.7, label='Smoothed')
        ax[1].legend()
        ax[1].set_title('Delta rotation angle = f(t)')

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas.tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


def update_graph(video, excluded_data, gaussian_sigma: float, spl_smoothing_factor: float, ax, canvas):
    (all_xs, raw_cumul_angle, smoothed_cumul_angle, raw_delta_angles,
     smoothed_delta_angles) = smooth_data_to_plot(video, excluded_data, gaussian_sigma, spl_smoothing_factor)

    ax[0].clear()
    ax[1].clear()

    ax[0].plot(all_xs, raw_cumul_angle, alpha=.7, label='Raw')
    ax[0].plot(all_xs, smoothed_cumul_angle, alpha=.7, label='Smoothed')
    ax[0].legend()
    ax[0].set_title('Cumulative rotation angle = f(t)')

    ax[1].plot(all_xs, raw_delta_angles, alpha=.7, label='Raw')
    ax[1].plot(all_xs, smoothed_delta_angles, alpha=.7, label='Smoothed')
    ax[1].legend()
    ax[1].set_title('Delta rotation angle = f(t)')
    canvas.draw()


def smooth_data_to_plot(video, excluded_data: str, gaussian_sigma: float, spl_smoothing_factor: float, ):
    raw_delta_angles = (np.array(video.angles)[1:] - np.array(video.angles)[:-1]).astype(np.float64)
    gaussian_delta_angles = gaussian_filter1d(raw_delta_angles, sigma=float(gaussian_sigma))

    all_xs = range(len(gaussian_delta_angles))

    truncated_delta_angles = gaussian_delta_angles.copy()

    if len(excluded_data.split(',')) >= 2:
        excluded_data = list(map(int, excluded_data.split(',')))
        for lim, next_lim in zip(excluded_data[::2], excluded_data[1::2]):
            truncated_delta_angles[lim:next_lim] = np.nan

    truncated_x_values = np.argwhere(~np.isnan(truncated_delta_angles))
    truncated_delta_angles = truncated_delta_angles[~np.isnan(truncated_delta_angles)]

    smoothing_spl = UnivariateSpline(truncated_x_values, truncated_delta_angles)
    smoothing_spl.set_smoothing_factor(float(spl_smoothing_factor))

    raw_cumul_angle = np.cumsum(raw_delta_angles)
    smoothed_cumul_angle = np.cumsum(smoothing_spl(all_xs))
    video.smoothed_angles = np.concatenate((np.array([video.angles[0]]),
                                            smoothed_cumul_angle + video.angles[0])).astype(np.float64)
    return all_xs, raw_cumul_angle, smoothed_cumul_angle, raw_delta_angles, smoothing_spl(all_xs)

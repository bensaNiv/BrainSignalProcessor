"""
This file was created by Niv Ben Salmon, Citri Lab, ELSC at the Hebrew University.
"""
from moviepy.editor import *
from Signal import *


class VideoPeaks:
    """
    Handles the parsing of a signal, creates the peaks object and the relevant time in the video
    """

    def __init__(self, signal: Signal, folder_name: str, video_path_side_cam1, video_path_side_cam2,
                 video_path_top_cam) -> None:
        """Gets ready to parse the input file.

        Args:
            input_file (typing.TextIO): input file.
        """
        self.signal = signal
        self.video_path_side_cam1 = folder_name + video_path_side_cam1
        self.video_path_side_cam2 = folder_name + video_path_side_cam2
        self.video_path_top_cam = folder_name + video_path_top_cam
        self.peaks_video_folder_name = self.create_name_peaks_folder()
        self.side_cam1_folder, self.side_cam2_folder, self.top_cam_folder = self.create_camera_peak_folders()
        self.clip_side_cam1 = VideoFileClip(self.video_path_side_cam1)
        self.clip_side_cam2 = VideoFileClip(self.video_path_side_cam2)
        self.clip_top_cam = VideoFileClip(self.video_path_top_cam)

    def create_videos_from_peak(self):
        for i, peak_coor in enumerate(self.signal.peaks):
            string_describing_peak = "peak" + str(i) + "_event.mp4"
            cut = int(10 * self.signal.frame_rate)
            peak_coor = self.signal.z_score[peak_coor]
            start = int(peak_coor) - cut if int(peak_coor) - cut > 0 else 0
            end = int(peak_coor) + cut if int(peak_coor) + cut < len(self.signal.z_score) else len(self.signal.z_score) - 1
            start_video = int(self.signal.time_vec[start]) - 5 if int(self.signal.time_vec[start]) - 5 >= 0 else 0
            end_video = int(self.signal.time_vec[end]) + 5 if int(self.signal.time_vec[end]) + 5 <= int(self.signal.time_vec[-1]) + 1 else int(self.signal.time_vec[-1]) + 1
            peak_clip_side_cam1 = self.clip_side_cam1.subclip(start_video, end_video)
            peak_clip_side_cam1.write_videofile(os.path.join(self.side_cam1_folder, string_describing_peak))
            peak_clip_side_cam2 = self.clip_side_cam1.subclip(start_video, end_video)
            peak_clip_side_cam2.write_videofile(os.path.join(self.side_cam2_folder, string_describing_peak))
            peak_clip_top_cam = self.clip_side_cam1.subclip(start_video, end_video)
            peak_clip_top_cam.write_videofile(os.path.join(self.top_cam_folder, string_describing_peak))

    def plot_signal_with_peaks(self):
        plt.plot(self.signal.time_vec, self.signal.z_score, "-", linewidth=0.5)
        plt.plot(self.signal.time_vec, np.zeros(len(self.signal.time_vec)), '--', color="olive")
        plt.plot(self.signal.time_vec, self.signal.peaks_bound, ':', color="red")
        plt.plot(self.signal.time_vec[self.signal.peaks], self.signal.z_score[self.signal.peaks], "x", color="orange")
        plt.xticks(range(int(self.signal.time_vec[-1]) + 1))
        plt.grid(True, linewidth=0.5, color='grey', linestyle='--')
        plt.xlabel("Time (min)")
        plt.ylabel("Z Score")
        plt.title(os.path.basename(self.signal.path_file) + " Normalized DF/F ")
        plt.tight_layout()
        plt.savefig(self.peaks_video_folder_name + "\\" + " peaks_plot_over_time" + ".png")
        plt.show()

    def create_name_peaks_folder(self):
        '''
        Create new folder for the peaks sub videos
        :return:
        '''
        peaks_video_folder_name = os.path.join(self.signal.result_dir, 'Peaks')
        if not os.path.isdir(peaks_video_folder_name):
            os.makedirs(peaks_video_folder_name)
        return peaks_video_folder_name

    def create_camera_peak_folders(self):
        ans = []
        for string in ("sideCam1", "sideCam2", "topCam"):
            camera_folder_name = os.path.join(self.peaks_video_folder_name, string)
            ans.append(camera_folder_name)
            if not os.path.isdir(camera_folder_name):
                os.makedirs(camera_folder_name)
        return ans

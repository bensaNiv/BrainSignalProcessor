import sys
import os
from Signal import Signal
# from VideoPeaks import VideoPeaks
import datetime
import re

DEFAULT_TAGS_PER_SECOND = 15

PATTERN_ISO = re.compile(r'(415)')

PATTERN_470 = re.compile(r'(470)')

PATTERN_560 = re.compile(r'(560)')

PATTEN_SIDE_CAM1 = re.compile(r'(sideCam1)')

PATTEN_SIDE_CAM2 = re.compile(r'(sideCam2)')

PATTEN_TOP_CAM = re.compile(r'(topCam)')


def adapt_video_to_name(pattern1, pattern2, pattern3, video_list):
    ans = ['', '', '']
    for file in video_list:
        if re.search(pattern1, file):
            ans[0] = file
            continue
        if re.search(pattern2, file):
            ans[1] = file
            continue
        if re.search(pattern3, file):
            ans[2] = file
            continue
    return ans


def process_folder(local_path, isos_name, filename1, filename2):
    poly_deg = 3
    baseline_window_in_minute = 2
    signals_array = [None, None, None, None]
    results_dir = os.path.join(local_path, 'Results ' + str(datetime.datetime.now()).replace(":", "-"))
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    # if filename2 != "\\":
    #     for i, region in enumerate(["Region0R", "Region1R"]):
    #         sig1 = Signal(local_path, filename2, region, isos_name, poly_deg, results_dir)
    #         signals_array[i] = sig1
    #         sig1.plot_figures_controler(0, 31, (1, 1, 1), False)
    if filename1 != "\\":
        for i, region in enumerate(["Region2G", "Region3G"]):
            sig2 = Signal(local_path, filename1, region, isos_name, results_dir, float('inf'), DEFAULT_TAGS_PER_SECOND,
                          poly_deg,
                          baseline_window_in_minute)
            signals_array[i + 2] = sig2
            sig2.plot_figures_controler(0, 31, (1, 1, 1), True)
    return signals_array


def parse_folder(_folder_path, csv_files_to_translate, _video_files):
    isos_name, filename1, filename2 = adapt_video_to_name(PATTERN_ISO, PATTERN_470, PATTERN_560, csv_files_to_translate)

    side_cam1, side_cam2, top_cam_video = adapt_video_to_name(PATTEN_SIDE_CAM1, PATTEN_SIDE_CAM2, PATTEN_TOP_CAM,
                                                              _video_files)

    left_r_signal, right_r_signal, left_g_signal, right_g_signal = \
        process_folder(_folder_path, "\\" + isos_name, "\\" + filename1, "\\" + filename2)

    # TODO : find from the folder the video and forward its path
    # TODO : Debug this section
    # if should_output_peak_videos:
        # # Create an VideoPeaks Object and save the peak and plot them
        # left_hem_peaks = VideoPeaks(left_g_signal, folder_path, "\\" + top_cam_video, "\\" + side_cam1,
        #                             "\\" + side_cam2)
        # right_hem_peaks = VideoPeaks(right_g_signal, folder_path, "\\" + top_cam_video, "\\" + side_cam1,
        #                              "\\" + side_cam2)
        #
        # left_hem_peaks.create_videos_from_peak()
        # left_hem_peaks.plot_signal_with_peaks()
        #
        # # right_hem_peaks.create_videos_from_peak()
        # right_hem_peaks.plot_signal_with_peaks()


if "__main__" == __name__:
    # Parses the input path and calls parse_file on each input file.
    # Parse all of the folder given.

    if not len(sys.argv) == 2:
        sys.exit("Invalid usage, please use: FolderParser <input path>")
    argument_path = os.path.abspath(sys.argv[1])
    if os.path.isdir(argument_path):
        folders_to_translate = [
            os.path.join(argument_path, filename)
            for filename in os.listdir(argument_path) if "Results" not in filename]
    else:
        folders_to_translate = [argument_path]
    should_output_peak_videos = False
    # each mouse in the folder
    for folder_path in folders_to_translate:
        # Needs to be a directory
        if os.path.isdir(folder_path):
            video_files = [file for file in os.listdir(folder_path) if os.path.splitext(file)[1] in (".avi", ".mp4")]
            files_to_translate = [file for file in os.listdir(folder_path) if os.path.splitext(file)[1] == ".csv"]
            parse_folder(folder_path, files_to_translate, video_files)

"""
This file was created by Niv Ben Salmon, Citri Lab, ELSC at the Hebrew University.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.signal import find_peaks
import re

from scipy.signal import savgol_filter
from statsmodels import robust

FRAME10 = 10
FRAME13 = 80 / 6
FRAME40 = 40
FRAME15 = 15


class Signal:
    """
    Handles the parsing of a signal, read the request data from the csv file and create relevant object
    """

    def __init__(self, path_file, input_file: str, region, isos_name, result_dir, tags_array_len, tags_per_second=15,
                 poly_degree=3,
                 baseline_window_in_minute=2) -> None:
        """Gets ready to parse the input file.

        Args:
            input_file (typing.TextIO): input file.
        """
        self.path_file = path_file
        self.input_file = input_file
        self.region = region
        self.isos_name = isos_name
        self.poly_degree = poly_degree
        self.frame_rate = 0
        self.result_dir = result_dir
        self.num_frames_in_array = 0
        self.baseline_window_in_minute = baseline_window_in_minute
        self.tags_array_len, self.tags_per_second = tags_array_len, tags_per_second
        self.time_vec, self.signal_vec, self.isos_vec = self.manipulation()
        self.z_score, self.df_f, self.control = self.create_z_score()
        self.file_name_save = self.create_file_name()
        # self.peaks, self.peaks_info, self.peaks_bound = self.find_peaks_for_signal()

    '''
    Manipulation Methods
    '''

    def manipulation(self):
        df_signal = self.create_df(self.input_file)
        df_isosbestic = self.create_df(self.isos_name)
        signal = self.create_signal_vec(df_signal)
        signal_isosbestic = self.create_signal_vec(df_isosbestic)
        time_vector = self.create_time_vec(df_signal)
        len_df = min(len(signal), len(signal_isosbestic))
        time_vector, signal, signal_isosbestic = map(lambda x: x[:len_df], [time_vector, signal,
                                                                            signal_isosbestic])
        time, signal, signal_isosbestic = map(lambda data_frame: np.array(data_frame).reshape(data_frame.shape),
                                              [time_vector, signal, signal_isosbestic])
        return time, signal, signal_isosbestic

    def create_z_score(self):
        baseline_window_size = self.calc_baseline_window_size_according_to_min(self.baseline_window_in_minute)

        isosbestic_coeff = np.polyfit(self.isos_vec.flatten(), self.signal_vec, deg=1)
        isosbestic_fitted = np.polyval(isosbestic_coeff, self.isos_vec)
        shape_of_np_arr = isosbestic_fitted.shape[0]
        window_filter = int(shape_of_np_arr * 0.002)
        window_length = window_filter if window_filter % 2 == 1 else window_filter + 1
        isosbesctic_signal_smooth = savgol_filter(np.array(isosbestic_fitted).reshape(shape_of_np_arr),
                                                  window_length=window_length,
                                                  polyorder=self.poly_degree)
        signal_smooth = savgol_filter(np.array(self.signal_vec).reshape(shape_of_np_arr),
                                      window_length=window_length,
                                      polyorder=self.poly_degree + 1)

        delta_f_over_f = np.divide(signal_smooth - isosbesctic_signal_smooth, isosbesctic_signal_smooth)

        # Sliding window on the filtered signal, create a locality in events
        delta_f_over_f = self.normalized_signal_with_baseline_window(delta_f_over_f, baseline_window_size)
        z_score = stats.zscore(delta_f_over_f)
        return z_score, delta_f_over_f, isosbesctic_signal_smooth

    def robust_z(self):
        # Robust Z score
        med = np.median(self.z_score)
        mad = robust.mad(self.z_score)[0]
        return np.divide(self.z_score - med, mad)

    '''
    Plot Methods
    '''

    def plot_figures_controler(self, start_time, end_time, fif: tuple, save=False):
        # start_time, end_time = self.dave_time_convert(0.5), self.dave_time_convert(1.5)

        start_index, end_index = self.convert_time_to_index(start_time, end_time)
        assert start_index < end_index, "Duration Time should be Positive"

        start_index = 0 if start_index < 0 else start_index
        end_index = self.time_vec.shape[0] if end_index > self.time_vec.shape[0] else end_index
        if fif[0]:
            self.plot_isosbestic(start_index, end_index, save)
        if fif[1]:
            self.plot_z_score(start_index, end_index, save)
        if fif[2]:
            self.plot_combine(start_index, end_index, save)

    def plot_isosbestic(self, start, end, save):
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        fig.set_figwidth(11)
        fig.set_figheight(8)
        time, sig, signal_isos = map(lambda x: x[start: end, :], [self.time_vec, self.signal_vec,
                                                                  self.isos_vec])
        control = self.control[start:end]
        ax1.plot(time, sig, "-", label="signal")
        ax1.plot(time, signal_isos, '--', color="red", label="isos_signal")
        ax1.legend()

        ax2.plot(time, sig, "-", label="signal")
        ax2.plot(time, control, '--', color="red", label="control")
        ax2.legend()

        plt.xlabel("Time (min) ")
        plt.suptitle("Signal vs the isosbastic and Control, " + self.region)
        plt.tight_layout()
        if save:
            plt.savefig(self.result_dir + "\\" + self.file_name_save + " control" + ".png")
        plt.show()

    def plot_combine(self, start, end, save):
        plot1 = plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=2)
        plot2 = plt.subplot2grid((4, 4), (0, 2), colspan=2, rowspan=2)
        plot3 = plt.subplot2grid((4, 4), (2, 0), colspan=4, rowspan=2)

        time, signal = map(lambda x: x[start: end, :], [self.time_vec, self.signal_vec])
        control, z_score = map(lambda x: x[start: end], [self.control, self.z_score])

        plot1.plot(time, signal, '-', color="grey")
        plot1.set_title("Signal, " + self.region)

        plot2.plot(time, control, color="orange")
        plot2.set_title(f"Scaled Control, polynomial from degree {self.poly_degree}", size=8)

        plot3.plot(time, z_score, '-')
        plot3.plot(time, np.zeros(len(time)), '--', color="red")
        plot3.set_title("Z_score")
        plt.xlabel("Time (min) ")

        plt.tight_layout()
        if save:
            plt.savefig(self.result_dir + "\\" + self.file_name_save + " combine" + ".png")
        plt.show()

    def plot_z_score(self, start, end, save):
        head_files = self.result_dir + "\\" + self.file_name_save + " z_score"
        time = (self.time_vec[start:end, :]).flatten()
        control, z_score = map(lambda x: x[start: end], [self.control, self.z_score])
        f = plt.figure()
        f.set_figwidth(20)
        f.set_figheight(4)
        plt.plot(time, z_score, "-", linewidth=1)
        plt.plot(time, np.zeros(len(time)), '--', color="olive")
        plt.xlabel("Time (min)")
        plt.ylabel("Z Score")
        plt.title(
            "Normalized DF/F " + ", " + self.region + " Polynomial degree of: " + str(self.poly_degree))
        plt.tight_layout()
        if save:
            np.savetxt(head_files + ".csv", np.dstack((time, control, z_score)).reshape(len(time), 3), "%f,%f,%f",
                       header="Time, Control, Z_Score")
            plt.savefig(head_files + ".png")
        plt.show()

    '''
    Help Functions
    '''

    def convert_time_to_index(self, start, stop):
        return int(self.frame_rate * 60 * start), int(self.frame_rate * 60 * stop)

    def create_signal_vec(self, df):
        try:
            return df[[self.region]]
        except Exception:
            assert "No region is detected"

    def set_frame_rate(self, gap):
        if 0.06 <= gap <= 0.07:
            self.frame_rate = FRAME15
        elif 0.07 <= gap <= 0.08:
            self.frame_rate = FRAME13
        elif 0.095 <= gap <= 0.15:
            self.frame_rate = FRAME10
        else:
            self.frame_rate = FRAME40

    def create_df(self, file):
        try:
            # return the data and without the first 30 sec
            df = pd.read_csv(self.path_file + file).set_index('FrameCounter')
            gap = df[["Timestamp"]].iloc[1][0] - df[["Timestamp"]].iloc[0][0]
            self.set_frame_rate(gap)
            if self.frame_rate == self.tags_per_second:
                num_frames = min(len(df), self.tags_array_len)
                df = df.iloc[:num_frames]
                self.num_frames_in_array = num_frames
                # Taking the first 30 seconds from the beginning  data frame
                cut = int(30 * self.frame_rate)
            else:
                common_divider = Signal.gcd(self.frame_rate, self.tags_per_second)
                reduce_signal_factor = self.frame_rate // common_divider
                self.frame_rate = common_divider
                reduce_tags_factor = self.tags_per_second // common_divider
                num_frames_after_refactor = min(len(df) // reduce_signal_factor,
                                                self.tags_array_len // reduce_tags_factor)
                self.num_frames_in_array = num_frames_after_refactor
                num_frames = num_frames_after_refactor * reduce_signal_factor
                df = df.iloc[:num_frames]
                df = df.iloc[::reduce_signal_factor]
                cut = int(30 * common_divider)
            return df.iloc[cut:]

        except FileNotFoundError:
            assert "File Not Found"

    def calc_baseline_window_size_according_to_min(self, minute):
        return self.frame_rate * 60 * minute

    def create_file_name(self):
        side = "Right" if "0" in self.region or "2" in self.region else "Left"
        mode = ''
        arr = re.split(r"[ /_]+", self.input_file[1:])
        date = ''
        for elm in arr:
            if "560" in elm:
                mode = "L560"
            elif "470" in elm:
                mode = "L470"
        return arr[0] + " " + side + " Side" + " " + mode + " " + date

    def fix_steps_(self):
        signal = self.signal_vec
        diff = np.diff(signal)

        diff_idx = diff.argsort()[::-1]
        start_ind, end_ind = diff_idx[:2]

        signal_cut_bad = signal[start_ind:end_ind + 1]
        x_cut_end = np.linspace(start_ind, end_ind, len(signal_cut_bad))

        z_bad = np.polyfit(x_cut_end, signal_cut_bad, 3)
        p_bad = np.poly1d(z_bad)
        fit_bad = p_bad(x_cut_end)
        fit_bad -= fit_bad[0] - signal[start_ind]

        signal[start_ind:end_ind] -= np.max(abs(fit_bad - signal[start_ind]))
        signal[end_ind:] -= (abs(fit_bad[-1] - signal[end_ind]))

        plt.figure(figsize=(11, 6))
        plt.plot(x_cut_end, fit_bad, color="red", label="fitted_bad")
        plt.plot(signal, color="purple", label="signal")
        plt.legend(loc="best")
        plt.show()

    def find_peaks_for_signal(self):
        """
        Method that find the peaks for the signal - according to Jessie
        #TODO find out who is that Jessie (Took it from some peaks paper- ask David)
        :return:
        """
        z_score_one_dimensional = self.z_score
        median = np.median(z_score_one_dimensional)
        series = pd.Series(z_score_one_dimensional)
        mad = series.mad()
        bound = np.linspace(median + 2 * mad, median + 2 * mad, len(z_score_one_dimensional))

        peaks, val_peaks = find_peaks(z_score_one_dimensional, height=bound, distance=int(13.3))

        median_peaks = np.median(val_peaks["peak_heights"])
        bound = np.linspace(median_peaks, median_peaks, len(z_score_one_dimensional))
        peaks, val_peaks = find_peaks(z_score_one_dimensional, height=bound, distance=int(self.frame_rate * 15))

        return peaks, val_peaks, bound

    def normalized_signal_with_baseline_window(self, delta_f_over_f, baseline_window_size):
        new_delta_f_over_f = np.copy(delta_f_over_f)
        range_window_size = baseline_window_size // 2
        for i in range(len(self.signal_vec)):
            window_start = int(np.max([0, i - range_window_size]))
            window_end = int(np.min([len(delta_f_over_f), i + range_window_size]))
            new_delta_f_over_f[i] = new_delta_f_over_f[i] - np.mean(delta_f_over_f[window_start:window_end])
        return new_delta_f_over_f

    @staticmethod
    def create_time_vec(df):
        # adding 0.5 because of the first half minute of the video that was cut
        return ((df[["Timestamp"]] - df[["Timestamp"]].iloc[0]) / 60) + 0.5

    @staticmethod
    def add_intercept(df):
        df.insert(0, "intercept", np.ones(len(df)))

    @staticmethod
    def gcd(a, b):
        """
        Finding the greates common divider using Euclid's algorithm
        """
        while b:
            a, b = b, a % b
        return a

    @staticmethod
    def dave_time_convert(x):
        return ((x * 10 ** 4) / 30) / 60

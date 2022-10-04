from functools import reduce

from matplotlib.colors import ListedColormap

import ScoreParser
from MouseExperiment import *


class Experiment:
    """
    Class the represent an experiment object
    """

    FRAMES_NOT_SET = -1

    def __init__(self, exp_path: str, manual_tag=0):
        self.path_to_experiment = exp_path
        self.mice_in_experiment = []
        self.plots_folder = self.create_plot_folder()
        self.not_left_right_exp = False
        self.mice_key_dict = self.parse_mice_key()
        # Intial value, the new value should be set by the first mouse added to the expirement
        self.manual_tag = manual_tag
        self.behaviors_container = {}
        self.frames_per_second_expiriment = self.FRAMES_NOT_SET
        self.expirement_score_parser = ScoreParser.ScoreParser()
        self.photometry_array_left, self.photometry_array_right = {}, {}
        self.photometry_array_opt1, self.photometry_array_opt2 = {}, {}

    def add_mouse(self, mouse: Mouse):
        """
        add new mouse to the experiment
        """
        self.mice_in_experiment.append(mouse)

    def get_num_mice_in_experiment(self):
        """
        return the number of mice in the experiment
        """
        return len(self.mice_in_experiment)

    def create_plot_folder(self):
        """
        Create an updated result folder for this expiremnent per run
        :return: the path to the plots folder
        """
        plot_dir_expe = os.path.join(self.path_to_experiment, 'plots')
        plot_dir_expe = os.path.join(plot_dir_expe,
                                     'Plots Experiment ' + str(datetime.datetime.now()).replace(":", "-")[:-10])
        if not os.path.isdir(plot_dir_expe):
            os.makedirs(plot_dir_expe)
        return plot_dir_expe

    def parse_mice_key(self):
        """
        parse a csv of the mice
        """
        key_dict = {}
        mice_key_file = os.path.join(self.path_to_experiment, 'mice_key.csv')
        df = pd
        if not os.path.isfile(mice_key_file):
            return key_dict
        try:
            df = pd.read_csv(mice_key_file)
        except Exception:
            assert "No mice_key.csv file was found"
        self.not_left_right_exp = True
        key_dict["values"] = {"opt1": df.iloc[[0]][["2G"]].iloc[0][0], "opt2": df.iloc[[0]][["3G"]].iloc[0][0]}
        for i, mice in enumerate(df["name"]):
            key_dict[mice] = {"2G": "", "3G": ""}
            key_dict[mice]["2G"] = df.iloc[[i]][["2G"]].iloc[0][0]
            key_dict[mice]["3G"] = df.iloc[[i]][["3G"]].iloc[0][0]
        return key_dict

    def set_frames_in_expiriment(self, num_frames):
        """
        Sets the frames per second in the expiernce.
        :param num_frames: the new number of frames.
        """
        self.frames_per_second_expiriment = num_frames

    def analyze_photometry_traces_bout_different_hemispheres(self, z_score_window_start, z_score_window_end,
                                                             normalize_window_start,
                                                             normalize_window_end,
                                                             behavior_time_seperation,
                                                             onset=True):
        """
        Inner function for an expirement that analyze the pgotometry signal in an onset bout for each hemisphere
        :return: A dictionary which for every behavior has a matrix of all the z_score bouts in all mice
        """
        reduce_from_start = z_score_window_start * self.frames_per_second_expiriment
        add_to_end = z_score_window_end * self.frames_per_second_expiriment
        normalize_window_start = normalize_window_start * self.frames_per_second_expiriment
        normalize_window_end = normalize_window_end * self.frames_per_second_expiriment
        photometry_arrays_left = {}
        photometry_arrays_right = {}

        for behavior_tag in self.behaviors_container:
            photometry_arrays_left[behavior_tag] = {'long': [], 'short': []}
            photometry_arrays_right[behavior_tag] = {'long': [], 'short': []}
            for mouse in self.mice_in_experiment:
                name = mouse.name
                left_z_score, right_z_score = mouse.left_signal.z_score, mouse.right_signal.z_score
                np_bouts, wp_bouts = self.expirement_score_parser. \
                    find_bouts_long_short_np_wp(mouse.tags_array, behavior_tag, 2, behavior_time_seperation,
                                                self.frames_per_second_expiriment, self.manual_tag,
                                                threshold_for_bout_length=1)

                for length in ['short', 'long']:
                    np_bouts_length = np_bouts[length]
                    for bout_idx in range(len(np_bouts_length)):
                        bout_array = np_bouts_length[bout_idx]
                        first_index_in_bout = bout_array[0]
                        last_index_in_bout = bout_array[-1]

                        start_index = first_index_in_bout if onset else last_index_in_bout
                        # check if the bout that was found fit the properties of size
                        if start_index >= reduce_from_start and left_z_score.shape[0] >= start_index + add_to_end:
                            sig_left = Experiment.parameterize_z_score(left_z_score, start_index,
                                                                       reduce_from_start, add_to_end,
                                                                       normalize_window_start, normalize_window_end,
                                                                       onset)
                            sig_right = Experiment.parameterize_z_score(right_z_score, start_index,
                                                                        reduce_from_start, add_to_end,
                                                                        normalize_window_start, normalize_window_end,
                                                                        onset)
                            if self.not_left_right_exp:
                                mouse_dict = self.mice_key_dict[name]
                                opt1, opt2 = self.mice_key_dict["values"]["opt1"], self.mice_key_dict["values"]["opt2"]
                                # We count as phtometry_array_left  as photometry array opt 1
                                if mouse_dict["2G"] == opt1:
                                    photometry_arrays_left[behavior_tag][length].append(sig_left)
                                    photometry_arrays_right[behavior_tag][length].append(sig_right)
                                else:
                                    photometry_arrays_left[behavior_tag][length].append(sig_right)
                                    photometry_arrays_right[behavior_tag][length].append(sig_left)
                            else:
                                photometry_arrays_left[behavior_tag][length].append(sig_left)
                                photometry_arrays_right[behavior_tag][length].append(sig_right)

        # At this point we have the photometry_arrays which has for every behavior an matrix of all the z_score bouts
        # in all mice
        if self.not_left_right_exp:
            self.photometry_array_opt1, self.photometry_array_opt2 = photometry_arrays_left, photometry_arrays_right
            self.photometry_array_left, self.photometry_array_right = None, None
        else:
            self.photometry_array_opt1, self.photometry_array_opt2 = None, None
            self.photometry_array_left, self.photometry_array_right = photometry_arrays_left, photometry_arrays_right

        return photometry_arrays_left, photometry_arrays_right

    def analyze_photometry_traces_bout_both_hemispheres(self, z_score_window_start, z_score_window_end,
                                                        normalize_window_start,
                                                        normalize_window_end,
                                                        behavior_time_seperation,
                                                        onset=True):
        """
        Inner function for an expirement that analyze the photometry signal in an onset bout
        :return: A dictionary which for every behavior has a matrix of all the z_score bouts in all mice
        """
        reduce_from_start = z_score_window_start * self.frames_per_second_expiriment
        add_to_end = z_score_window_end * self.frames_per_second_expiriment
        normalize_window_start = normalize_window_start * self.frames_per_second_expiriment
        normalize_window_end = normalize_window_end * self.frames_per_second_expiriment
        photometry_arrays = {}

        for behavior_tag in self.behaviors_container:
            photometry_arrays[behavior_tag] = {'long': [], 'short': []}
            for mouse in self.mice_in_experiment:
                left_z_score, right_z_score = mouse.left_signal.z_score, mouse.right_signal.z_score
                np_bouts, wp_bouts = self.expirement_score_parser. \
                    find_bouts_long_short_np_wp(mouse.tags_array, behavior_tag, 2, behavior_time_seperation,
                                                self.frames_per_second_expiriment, self.manual_tag,
                                                threshold_for_bout_length=1)

                for length in ['short', 'long']:
                    np_bouts_length = np_bouts[length]
                    for bout_idx in range(len(np_bouts_length)):
                        bout_array = np_bouts_length[bout_idx]
                        first_index_in_bout = bout_array[0]
                        last_index_in_bout = bout_array[-1]

                        start_index = first_index_in_bout if onset else last_index_in_bout
                        # check if the bout that was found fit the properties of size
                        if start_index >= reduce_from_start and left_z_score.shape[0] >= start_index + add_to_end:
                            sig_left = Experiment.parameterize_z_score(left_z_score, start_index,
                                                                       reduce_from_start, add_to_end,
                                                                       normalize_window_start, normalize_window_end,
                                                                       onset)
                            sig_right = Experiment.parameterize_z_score(right_z_score, start_index,
                                                                        reduce_from_start, add_to_end,
                                                                        normalize_window_start, normalize_window_end,
                                                                        onset)

                            photometry_arrays[behavior_tag][length].append(sig_left)
                            photometry_arrays[behavior_tag][length].append(sig_right)

        # At this point we have the photometry_arrays which has for every behavior an matrix of all the z_score bouts
        # in all mice
        return photometry_arrays

    def plot_signals_according_to_old_code(self, photometry_arrays, window_before, window_after, plot_mouse_cases=False,
                                           given_heading=""):
        for length in ['short', 'long']:
            for behavior_tag in self.behaviors_container:
                z_score_lst_all_mice_behavior = photometry_arrays[behavior_tag][length]
                if len(z_score_lst_all_mice_behavior) <= 0:
                    continue

                conc = np.stack(z_score_lst_all_mice_behavior, axis=0)
                sem_conc = np.std(conc, axis=0) / np.sqrt(len(conc))
                if self.manual_tag:
                    behavior = reduce(lambda x, y: x + " and " + y,
                                      map(lambda x: ACTION_DICT[x][1], set(behavior_tag)))
                else:
                    behavior = STEREO_DICT_REV[behavior_tag]

                title_to_save = "Normalized DF_F - " + given_heading + " " + length + " " + behavior + ", " + str(
                    len(conc)) + " events"
                title = title_to_save + " mean"

                window_start = window_before
                window_end = window_after
                xx = np.linspace(-window_start, window_end + 1, len(conc[0]))
                mean_all_mic = conc.mean(axis=0)
                max_val, min_val = max(mean_all_mic), min(mean_all_mic)
                if plot_mouse_cases:
                    self.plot_actions_before_and_after_bouts_onset(z_score_lst_all_mice_behavior, xx,
                                                                   "seperate " + title_to_save)
                plt.fill_between(xx, mean_all_mic - sem_conc, mean_all_mic + sem_conc, alpha=.5, linewidth=0,
                                 color="grey")
                plt.plot(xx, mean_all_mic, marker='', color='black', linewidth=2, alpha=0.7)
                plt.text(window_start + 0.7, mean_all_mic[-1], "mean", rotation=45, va='center', size='large',
                         color='black')
                plt.grid(True, linewidth=0.2, color='grey', linestyle='--', alpha=0.5)
                plt.xticks(range(-window_start, window_end + 1, 2))
                plt.xlabel("Time (sec)")
                plt.ylabel("Z Score")
                plt.ylim(min_val - 1, max_val + 1)
                plt.title(title)
                plt.savefig(self.plots_folder + "\\" + title_to_save + ".png")
                plt.show()

    def plot_actions_before_and_after_bouts_onset(self, photometry_arrays, x_line, title):
        f = plt.figure()
        f.set_figwidth(10)
        f.set_figheight(4)
        cmap = Experiment.get_cmapi(0)
        for i in range(len(photometry_arrays)):
            plt.plot(x_line, photometry_arrays[i], "-", linewidth=0.5, alpha=1, color=cmap(i))
        plt.grid(True, linewidth=0.2, color='grey', linestyle='--', alpha=0.5)
        plt.xlabel("Time (sec)")
        plt.ylabel("Z Score")
        plt.ylim(-5, 5)
        plt.title(title)
        plt.savefig(self.plots_folder + "\\" + title + ".png")
        plt.show()

    def plot_actions_together_for_each_hemisphere(self, derivative=False, window_before=10, window_after=20):
        """
        Function that plots all of the behaviors together (mean of each behavior for all mice in experiment),
        in the same figure there is representation for each hemisphere
        """
        for length in ['short', 'long']:
            f = plt.figure()
            f.set_figwidth(16)
            f.set_figheight(10)
            title_to_save = "All Behaviors - mean " + length
            if derivative:
                title_to_save = "Derivative " + title_to_save
            if self.not_left_right_exp:
                title_to_save = self.mice_key_dict["values"]["opt1"] + "-" + self.mice_key_dict["values"]["opt2"] + \
                                " " + title_to_save
            title = title_to_save
            cmap_param = 0 if not self.manual_tag else len(
                self.behaviors_container)  # Indicates how many and the types of colors for the behaviors plot

            cmap = Experiment.get_cmapi(cmap_param)
            for i, behavior_tag in enumerate(self.behaviors_container):
                if not self.manual_tag and behavior_tag in {'v', 'j'}:
                    continue

                if self.manual_tag:
                    behavior = reduce(lambda x, y: x + " and " + y,
                                      map(lambda x: ACTION_DICT[x][1], set(behavior_tag)))
                else:
                    behavior = STEREO_DICT_REV[behavior_tag]

                if self.not_left_right_exp:
                    tag1, tag2 = self.mice_key_dict["values"]["opt1"], self.mice_key_dict["values"]["opt2"]
                    z_score_lst_all_mice_behavior_left = self.photometry_array_opt1[behavior_tag][length]
                    z_score_lst_all_mice_behavior_right = self.photometry_array_opt2[behavior_tag][length]
                else:
                    tag1, tag2 = "left", "right"
                    z_score_lst_all_mice_behavior_left = self.photometry_array_left[behavior_tag][length]
                    z_score_lst_all_mice_behavior_right = self.photometry_array_right[behavior_tag][length]

                if z_score_lst_all_mice_behavior_left is None or z_score_lst_all_mice_behavior_right is None:
                    assert "Something wrong with the assigning of the photometry arrays"

                num_events_left = len(z_score_lst_all_mice_behavior_left)
                num_events_right = len(z_score_lst_all_mice_behavior_right)

                if len(z_score_lst_all_mice_behavior_right) <= 4:
                    continue
                conc_left = np.stack(z_score_lst_all_mice_behavior_left, axis=0)
                conc_right = np.stack(z_score_lst_all_mice_behavior_right, axis=0)

                window_start = window_before
                window_end = window_after
                xx = np.linspace(-window_start, window_end + 1, len(conc_left[0]))

                mean_all_mice_left = conc_left.mean(axis=0)
                mean_all_mice_right = conc_right.mean(axis=0)

                if derivative:
                    dxdy_left = np.diff(mean_all_mice_left) / np.diff(xx)
                    dxdy_right = np.diff(mean_all_mice_right) / np.diff(xx)
                    plt.plot(xx[:-1], dxdy_left, "-", color=cmap(i), linewidth=2, alpha=0.9,
                             label=behavior + " " + tag1 + " " + str(num_events_left) + " events")
                    plt.plot(xx[:-1], dxdy_right, "--", color=cmap(i), linewidth=2, alpha=0.9,
                             label=behavior + " " + tag2 + " " + str(num_events_right) + " events")
                else:
                    plt.plot(xx, mean_all_mice_left, "-", color=cmap(i),
                             linewidth=2, alpha=0.9,
                             label=behavior + " " + tag1 + " " + str(num_events_left) + " events")
                    plt.plot(xx, mean_all_mice_right, "--", color=cmap(i), linewidth=2, alpha=0.9,
                             label=behavior + " " + tag2 + " " + str(num_events_right) + " events")

            plt.grid(True, linewidth=0.2, color='grey', linestyle='--', alpha=0.5)
            plt.axvspan(-1, 2, color='red', alpha=0.2)
            plt.xticks(range(-window_before, window_after + 1, 2))
            plt.ylim(-5, 5)
            plt.xlabel("Time (sec)")
            plt.ylabel("Z Score")
            plt.legend(loc='best')
            plt.title(title)
            plt.savefig(self.plots_folder + "\\" + title_to_save + ".png")
            plt.show()

    def plot_actions_separately_for_each_hemisphere(self, derivative=False, window_before=10, window_after=20):
        """
        Function that plots each behavior (mean of each behavior for all mice in experiment),
        in a figure there is representation for each hemisphere
        """
        for length in ['short', 'long']:
            f = plt.figure()
            f.set_figwidth(16)
            f.set_figheight(10)
            cmap_param = 0 if not self.manual_tag else len(
                self.behaviors_container)  # Indicates how many and the types of colors for the behaviors plot

            cmap = Experiment.get_cmapi(cmap_param)
            for i, behavior_tag in enumerate(self.behaviors_container):
                if not self.manual_tag and behavior_tag in {'v', 'j'}:
                    continue

                if self.manual_tag:
                    behavior = reduce(lambda x, y: x + " and " + y,
                                      map(lambda x: ACTION_DICT[x][1], set(behavior_tag)))
                else:
                    behavior = STEREO_DICT_REV[behavior_tag]

                title_to_save = behavior + " " + length
                if derivative:
                    title_to_save = "Derivative " + title_to_save
                if self.not_left_right_exp:
                    title_to_save = self.mice_key_dict["values"]["opt1"] + "-" + self.mice_key_dict["values"]["opt2"]\
                                    + " " + title_to_save
                title = title_to_save

                if self.not_left_right_exp:
                    tag1, tag2 = self.mice_key_dict["values"]["opt1"], self.mice_key_dict["values"]["opt2"]
                    z_score_lst_all_mice_behavior_left = self.photometry_array_opt1[behavior_tag][length]
                    z_score_lst_all_mice_behavior_right = self.photometry_array_opt2[behavior_tag][length]
                else:
                    tag1, tag2 = "left", "right"
                    z_score_lst_all_mice_behavior_left = self.photometry_array_left[behavior_tag][length]
                    z_score_lst_all_mice_behavior_right = self.photometry_array_right[behavior_tag][length]

                if z_score_lst_all_mice_behavior_left is None or z_score_lst_all_mice_behavior_right is None:
                    assert "Something wrong with the assigning of the photometry arrays"

                num_events_left = len(z_score_lst_all_mice_behavior_left)
                num_events_right = len(z_score_lst_all_mice_behavior_right)

                if len(z_score_lst_all_mice_behavior_right) <= 0:
                    continue
                conc_left = np.stack(z_score_lst_all_mice_behavior_left, axis=0)
                conc_right = np.stack(z_score_lst_all_mice_behavior_right, axis=0)

                window_start = window_before
                window_end = window_after
                xx = np.linspace(-window_start, window_end + 1, len(conc_left[0]))

                mean_all_mice_left = conc_left.mean(axis=0)
                mean_all_mice_right = conc_right.mean(axis=0)

                if derivative:
                    dxdy_left = np.diff(mean_all_mice_left) / np.diff(xx)
                    dxdy_right = np.diff(mean_all_mice_right) / np.diff(xx)
                    plt.plot(xx[:-1], dxdy_left, "-", color=cmap(i), linewidth=2, alpha=0.9,
                             label=behavior + " " + tag1 + " " + str(num_events_left) + " events")
                    plt.plot(xx[:-1], dxdy_right, "--", color=cmap(i), linewidth=2, alpha=0.9,
                             label=behavior + " " + tag2 + " " + str(num_events_right) + " events")
                else:
                    plt.plot(xx, mean_all_mice_left, "-", color=cmap(i), linewidth=2, alpha=0.9,
                             label=behavior + " " + tag1 + " " + str(num_events_left) + " events")
                    plt.plot(xx, mean_all_mice_right, "--", color=cmap(i), linewidth=2, alpha=0.9,
                             label=behavior + " " + tag2 + " " + str(num_events_right) + " events")

                plt.grid(True, linewidth=0.2, color='grey', linestyle='--', alpha=0.5)
                plt.axvspan(-1, 2, color='red', alpha=0.2)
                plt.xticks(range(-window_before, window_after + 1, 2))
                plt.xlabel("Time (sec)")
                plt.ylabel("Z Score")
                plt.ylim(-5, 5)
                plt.legend(loc='best')
                plt.title(title)
                plt.savefig(self.plots_folder + "\\" + title_to_save + ".png")
                plt.show()

    def plot_actions_together_for_each_hemisphere_normalize_according_to_max_min(self, max_val, min_val,
                                                                                 behavior_to_normalize_according_to='',
                                                                                 window_before=10, window_after=20):
        """
        Function that plot the instances of the signals for each behavior. Normalized all the mean instances according
        to a certain behavior and max/min val. If a 'behavior_to_normalize_according_to' is not default then the sigals
        are normalized according to the max and min val of this behavior.
        Otherwise, its normalized according to the global max/min val.
        :param max_val: Max val to normnalie to
        :param min_val: Min val to normalize to
        :param behavior_to_normalize_according_to: Behavior that we normalize according to
        :param window_before: Time before to plot
        :param window_after:Time after to plot
        """
        window_start = window_before
        window_end = window_after
        tag1, tag2 = "", ""
        for length in ['short', 'long']:
            f = plt.figure()
            f.set_figwidth(16)
            f.set_figheight(10)
            title = ''
            title_to_save = "Normalized All Behaviors - mean " + length + " according to "

            max_left_to_normalize_with = float('-inf')
            min_left_to_normalize_with = float('inf')
            max_right_to_normalize_with = float('-inf')
            min_right_to_normalize_with = float('inf')

            conc_actions_dict_left = {}
            conc_actions_dict_right = {}
            max_actions_dict_left = {}
            max_actions_dict_right = {}
            min_actions_dict_left = {}
            min_actions_dict_right = {}
            num_events_dict_left = {}
            num_events_dict_right = {}

            cmap_param = 0 if not self.manual_tag else len(
                self.behaviors_container)  # Indicates how many and the types of colors for the behaviors plot

            cmap = Experiment.get_cmapi(cmap_param)
            for i, behavior_tag in enumerate(self.behaviors_container):
                if not self.manual_tag and behavior_tag in {'v', 'j'}:
                    continue

                if self.not_left_right_exp:
                    tag1, tag2 = self.mice_key_dict["values"]["opt1"], self.mice_key_dict["values"]["opt2"]
                    z_score_lst_all_mice_behavior_left = self.photometry_array_opt1[behavior_tag][length]
                    z_score_lst_all_mice_behavior_right = self.photometry_array_opt2[behavior_tag][length]
                else:
                    tag1, tag2 = "left", "right"
                    z_score_lst_all_mice_behavior_left = self.photometry_array_left[behavior_tag][length]
                    z_score_lst_all_mice_behavior_right = self.photometry_array_right[behavior_tag][length]

                if z_score_lst_all_mice_behavior_left is None or z_score_lst_all_mice_behavior_right is None:
                    assert "Something wrong with the assigning of the photometry arrays"

                num_events_left = len(z_score_lst_all_mice_behavior_left)
                num_events_right = len(z_score_lst_all_mice_behavior_right)

                if len(z_score_lst_all_mice_behavior_right) <= 4:
                    continue

                conc_left = np.stack(z_score_lst_all_mice_behavior_left, axis=0)
                conc_right = np.stack(z_score_lst_all_mice_behavior_right, axis=0)

                mean_all_mice_left = conc_left.mean(axis=0)
                mean_all_mice_right = conc_right.mean(axis=0)

                conc_actions_dict_left[behavior_tag] = mean_all_mice_left
                conc_actions_dict_right[behavior_tag] = mean_all_mice_right
                max_actions_dict_left[behavior_tag] = max(mean_all_mice_left)
                max_actions_dict_right[behavior_tag] = max(mean_all_mice_right)
                min_actions_dict_left[behavior_tag] = min(mean_all_mice_left)
                min_actions_dict_right[behavior_tag] = min(mean_all_mice_right)
                num_events_dict_left[behavior_tag] = num_events_left
                num_events_dict_right[behavior_tag] = num_events_right

            # at this point we have 2 dictionaries with all of the behaviors, their mean array for all mice in
            # experiment, and the max value for that behavior

            # finding max to normalize according to
            if behavior_to_normalize_according_to != '':
                title_to_save += behavior_to_normalize_according_to
                max_left_to_normalize_with = max_actions_dict_left[behavior_to_normalize_according_to]
                max_right_to_normalize_with = max_actions_dict_right[behavior_to_normalize_according_to]
            else:
                title = title_to_save + "max val: " + str(max_val) + ", min val: " + str(min_val)
                title_to_save += "max min"
                for i, behavior_tag in enumerate(self.behaviors_container):
                    if not self.manual_tag and behavior_tag in {'v', 'j'}:
                        continue
                    if behavior_tag not in max_actions_dict_left.keys():
                        continue
                    max_left_to_normalize_with = max(max_actions_dict_left[behavior_tag], max_left_to_normalize_with)
                    max_right_to_normalize_with = max(max_actions_dict_right[behavior_tag], max_right_to_normalize_with)
                    min_left_to_normalize_with = min(min_actions_dict_left[behavior_tag], min_left_to_normalize_with)
                    min_right_to_normalize_with = min(min_actions_dict_right[behavior_tag], min_right_to_normalize_with)

            for i, behavior_tag in enumerate(self.behaviors_container):
                if not self.manual_tag and behavior_tag in {'v', 'j'}:
                    continue
                if behavior_tag not in max_actions_dict_left.keys():
                    continue
                if self.manual_tag:
                    behavior = reduce(lambda x, y: x + " and " + y,
                                      map(lambda x: ACTION_DICT[x][1], set(behavior_tag)))
                else:
                    behavior = STEREO_DICT_REV[behavior_tag]

                mean_behavior_left_normalize = Experiment.normalize_array_according_to_max_min(
                    conc_actions_dict_left[behavior_tag], max_val, min_val, max_left_to_normalize_with,
                    min_left_to_normalize_with)
                mean_behavior_right_normalize = Experiment.normalize_array_according_to_max_min(
                    conc_actions_dict_right[behavior_tag], max_val, min_val, max_right_to_normalize_with,
                    min_right_to_normalize_with)
                num_events_left, num_events_right = num_events_dict_left[behavior_tag], num_events_dict_right[
                    behavior_tag]
                xx = np.linspace(-window_start, window_end + 1, len(mean_behavior_left_normalize))
                plt.plot(xx, mean_behavior_left_normalize, "-", color=cmap(i), linewidth=2, alpha=0.9,
                         label=behavior + " " + tag1 + " " + str(num_events_left) + " events")
                plt.plot(xx, mean_behavior_right_normalize, "--", color=cmap(i), linewidth=2, alpha=0.9,
                         label=behavior + " " + tag2 + " " + str(num_events_right) + " events")

            plt.grid(True, linewidth=0.2, color='grey', linestyle='--', alpha=0.5)
            plt.axvspan(-1, 2, color='red', alpha=0.2)
            plt.xticks(range(-window_before, window_after + 1, 2))
            plt.xlabel("Time (sec)")
            plt.ylabel("Z Score")
            plt.ylim(min_val - 0.25, max_val + 0.25)
            # plt.yticks(range(-0.25, 1.25, 0.25))
            plt.legend(loc='best')
            plt.title(title)
            plt.savefig(self.plots_folder + "\\" + title_to_save + ".png")
            plt.show()

    @staticmethod
    def parameterize_z_score(z_score, start_index_in_bout, reduce_from_start, add_to_end, normalize_window_start,
                             normalize_window_end, onset):
        """
        Return a normalized z_score according to the requested values
        """
        if onset:
            return z_score[start_index_in_bout - reduce_from_start:start_index_in_bout + add_to_end] - np.mean(
                z_score[start_index_in_bout - normalize_window_start:start_index_in_bout - normalize_window_end])
        return z_score[start_index_in_bout - reduce_from_start:start_index_in_bout + add_to_end] - np.mean(
            z_score[start_index_in_bout + normalize_window_start:start_index_in_bout + normalize_window_end])

    @staticmethod
    def get_cmapi(n, name='hsv'):
        """
        Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.
        """
        if not n:
            return ListedColormap(STEREO_COLOR_DICT.values())
        return plt.cm.get_cmap(name, n)

    @staticmethod
    def normalize_array_according_to_max_min(array, max_val, min_val, max_array, min_array):
        """
        Normalize an array accordin to max/min values
        """
        return (((array - min_array) / (max_array - min_array)) * (max_val - min_val)) + min_val

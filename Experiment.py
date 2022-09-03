import ScoreParser
from MouseExperiment import *


class Experiment:
    '''
    Class the represent an experiment object
    '''

    FRAME_PER_SECOND = 15

    def __init__(self, exp_path: str, dorsal_ventral_exp=False):
        self.path_to_experiment = exp_path
        self.mice_in_experiment = []
        self.plots_folder = self.create_plot_folder()
        self.expirement_score_parser = ScoreParser.ScoreParser()
        self.photometry_array_left, self.photometry_array_right = {}, {}

        # todo think about how I add mice to this folder - each time a mouse is created maybe?

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

    def analyze_photometry_traces_bout_different_hemispheres(self, z_score_window_start, z_score_window_end,
                                                             normalize_window_start,
                                                             normalize_window_end, onset=True):
        """
        Inner function for an expirement that analyze the pgotometry signal in an onset bout for each hemisphere
        :return: A dictionary which for every behavior has a matrix of all the z_score bouts in all mice
        """
        reduce_from_start = z_score_window_start * Experiment.FRAME_PER_SECOND
        add_to_end = z_score_window_end * Experiment.FRAME_PER_SECOND
        normalize_window_start = normalize_window_start * Experiment.FRAME_PER_SECOND
        normalize_window_end = normalize_window_end * Experiment.FRAME_PER_SECOND

        photometry_arrays_left = {}
        photometry_arrays_right = {}
        for behavior in STEREO_BEHAVIORS:
            behavior_tag = STEREO_DICT[behavior]
            photometry_arrays_left[behavior_tag] = {}
            photometry_arrays_right[behavior_tag] = {}
        for behavior in STEREO_BEHAVIORS:
            behavior_tag = STEREO_DICT[behavior]
            photometry_arrays_left[behavior_tag] = {'long': [], 'short': []}
            photometry_arrays_right[behavior_tag] = {'long': [], 'short': []}
            for mouse in self.mice_in_experiment:
                left_z_score, right_z_score = mouse.left_signal.z_score, mouse.right_signal.z_score
                np_bouts, wp_bouts = self.expirement_score_parser.find_bouts_long_short_np_wp(mouse.tags_array,
                                                                                              behavior_tag,
                                                                                              short_long_threshold=2)

                for length in ['short', 'long']:
                    np_bouts_length = np_bouts[length]
                    for bout_idx in range(len(np_bouts_length)):
                        bout_array = np_bouts_length[bout_idx]
                        first_index_in_bout = bout_array[0]
                        last_index_in_bout = bout_array[-1]

                        start_index = first_index_in_bout if onset else last_index_in_bout
                        # check if the bout that was found fit the properties of size
                        if start_index >= reduce_from_start and left_z_score.shape[
                            0] >= start_index + add_to_end:
                            sig_left = Experiment.parameterize_z_score(left_z_score, start_index,
                                                                       reduce_from_start, add_to_end,
                                                                       normalize_window_start, normalize_window_end,
                                                                       onset)
                            sig_right = Experiment.parameterize_z_score(right_z_score, start_index,
                                                                        reduce_from_start, add_to_end,
                                                                        normalize_window_start, normalize_window_end,
                                                                        onset)

                            photometry_arrays_left[behavior_tag][length].append(sig_left)
                            photometry_arrays_right[behavior_tag][length].append(sig_right)

        # At this point we have the photometry_arrays which has for every behavior an matrix of all the z_score bouts
        # in all mice
        self.photometry_array_left, self.photometry_array_right = photometry_arrays_left, photometry_arrays_right
        return photometry_arrays_left, photometry_arrays_right

    def analyze_photometry_traces_bout_both_hemispheres(self, z_score_window_start, z_score_window_end,
                                                        normalize_window_start,
                                                        normalize_window_end, onset=True):
        """
        Inner function for an expirement that analyze the photometry signal in an onset bout
        :return: A dictionary which for every behavior has a matrix of all the z_score bouts in all mice
        """
        reduce_from_start = z_score_window_start * Experiment.FRAME_PER_SECOND
        add_to_end = z_score_window_end * Experiment.FRAME_PER_SECOND
        normalize_window_start = normalize_window_start * Experiment.FRAME_PER_SECOND
        normalize_window_end = normalize_window_end * Experiment.FRAME_PER_SECOND

        photometry_arrays = {}
        for behavior in STEREO_BEHAVIORS:
            behavior_tag = STEREO_DICT[behavior]
            photometry_arrays[behavior_tag] = {}
        for behavior in STEREO_BEHAVIORS:
            behavior_tag = STEREO_DICT[behavior]
            photometry_arrays[behavior_tag] = {'long': [], 'short': []}
            for mouse in self.mice_in_experiment:
                left_z_score, right_z_score = mouse.left_signal.z_score, mouse.right_signal.z_score
                np_bouts, wp_bouts = self.expirement_score_parser.find_bouts_long_short_np_wp(mouse.tags_array,
                                                                                              behavior_tag,
                                                                                              short_long_threshold=2)

                for length in ['short', 'long']:
                    np_bouts_length = np_bouts[length]
                    for bout_idx in range(len(np_bouts_length)):
                        bout_array = np_bouts_length[bout_idx]
                        first_index_in_bout = bout_array[0]
                        last_index_in_bout = bout_array[-1]

                        start_index = first_index_in_bout if onset else last_index_in_bout
                        # check if the bout that was found fit the properties of size
                        if start_index >= reduce_from_start and left_z_score.shape[
                            0] >= start_index + add_to_end:
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
            for behavior in STEREO_BEHAVIORS:
                behavior_tag = STEREO_DICT[behavior]
                z_score_lst_all_mice_behavior = photometry_arrays[behavior_tag][length]
                if len(z_score_lst_all_mice_behavior) <= 0:
                    continue

                conc = np.stack(z_score_lst_all_mice_behavior, axis=0)
                sem_conc = np.std(conc, axis=0) / np.sqrt(len(conc))
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
        # todo continue from here
        f = plt.figure()
        f.set_figwidth(10)
        f.set_figheight(4)
        cmap = Experiment.get_cmapi(len(photometry_arrays))
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
            title = title_to_save

            cmap = Experiment.get_cmapi(len(STEREO_BEHAVIORS))
            for i, behavior in enumerate(STEREO_BEHAVIORS):
                behavior_tag = STEREO_DICT[behavior]
                if behavior_tag in {'v', 'j'}:
                    continue
                z_score_lst_all_mice_behavior_left = self.photometry_array_left[behavior_tag][length]
                z_score_lst_all_mice_behavior_right = self.photometry_array_right[behavior_tag][length]
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
                             label=behavior + " left")
                    plt.plot(xx[:-1], dxdy_right, "--", color=cmap(i), linewidth=2, alpha=0.9,
                             label=behavior + " right")
                else:
                    plt.plot(xx, mean_all_mice_left, "-", color=cmap(i), linewidth=2, alpha=0.9,
                             label=behavior + " left")
                    plt.plot(xx, mean_all_mice_right, "--", color=cmap(i), linewidth=2, alpha=0.9,
                             label=behavior + " right")

            plt.grid(True, linewidth=0.2, color='grey', linestyle='--', alpha=0.5)
            plt.xticks(range(-window_before, window_after + 1, 2))
            plt.xlabel("Time (sec)")
            plt.ylabel("Z Score")
            plt.ylim(-5, 5)
            plt.legend(loc='best')
            plt.title(title)
            plt.savefig(self.plots_folder + "\\" + title_to_save + ".png")
            plt.show()

    def plot_actions_together_for_each_hemisphere_normalize_according_to_max(self,
                                                                             behavior_to_normalize_according_to='',
                                                                             window_before=10, window_after=20):
        window_start = window_before
        window_end = window_after
        for length in ['short', 'long']:
            f = plt.figure()
            f.set_figwidth(16)
            f.set_figheight(10)
            title = "Normalized All Behaviors - mean " + length + " according to "
            cmap = Experiment.get_cmapi(len(STEREO_BEHAVIORS))

            max_right_to_normalize_with = float('-inf')
            max_left_to_normalize_with = float('-inf')

            conc_actions_dict_left = {}
            conc_actions_dict_right = {}
            max_actions_dict_left = {}
            max_actions_dict_right = {}

            for i, behavior in enumerate(STEREO_BEHAVIORS):
                behavior_tag = STEREO_DICT[behavior]
                if behavior_tag in {'v', 'j'}:
                    continue
                z_score_lst_all_mice_behavior_left = self.photometry_array_left[behavior_tag][length]
                z_score_lst_all_mice_behavior_right = self.photometry_array_right[behavior_tag][length]

                if len(z_score_lst_all_mice_behavior_right) <= 0:
                    continue

                conc_left = np.stack(z_score_lst_all_mice_behavior_left, axis=0)
                conc_right = np.stack(z_score_lst_all_mice_behavior_right, axis=0)

                mean_all_mice_left = conc_left.mean(axis=0)
                mean_all_mice_right = conc_right.mean(axis=0)

                conc_actions_dict_left[behavior_tag] = mean_all_mice_left
                conc_actions_dict_right[behavior_tag] = mean_all_mice_right
                max_actions_dict_left[behavior_tag] = max(mean_all_mice_left)
                max_actions_dict_right[behavior_tag] = max(mean_all_mice_right)

            # at this point we have 2 dictionaries with all of the behaviors, their mean array for all mice in
            # experiment, and the max value for that behavior

            # finding max to normalize according to
            if behavior_to_normalize_according_to != '':
                title += behavior_to_normalize_according_to
                max_left_to_normalize_with = max_actions_dict_left[behavior_to_normalize_according_to]
                max_right_to_normalize_with = max_actions_dict_right[behavior_to_normalize_according_to]
            else:
                title += "max val"
                for i, behavior in enumerate(STEREO_BEHAVIORS):
                    behavior_tag = STEREO_DICT[behavior]
                    if behavior_tag in {'v', 'j'} or behavior_tag not in max_actions_dict_left.keys():
                        continue
                    max_left_to_normalize_with = max(max_actions_dict_left[behavior_tag], max_left_to_normalize_with)
                    max_right_to_normalize_with = max(max_actions_dict_right[behavior_tag], max_right_to_normalize_with)

            for i, behavior in enumerate(STEREO_BEHAVIORS):
                behavior_tag = STEREO_DICT[behavior]
                if behavior_tag in {'v', 'j'} or behavior_tag not in max_actions_dict_left.keys():
                    continue
                mean_behavior_left_normalize = conc_actions_dict_left[behavior_tag] / max_left_to_normalize_with
                mean_behavior_right_normalize = conc_actions_dict_right[behavior_tag] / max_right_to_normalize_with
                xx = np.linspace(-window_start, window_end + 1, len(mean_behavior_left_normalize))
                plt.plot(xx, mean_behavior_left_normalize, "-", color=cmap(i), linewidth=2, alpha=0.9,
                         label=behavior + " left")
                plt.plot(xx, mean_behavior_right_normalize, "--", color=cmap(i), linewidth=2, alpha=0.9,
                         label=behavior + " right")


            plt.grid(True, linewidth=0.2, color='grey', linestyle='--', alpha=0.5)
            plt.xticks(range(-window_before, window_after + 1, 2))
            plt.xlabel("Time (sec)")
            plt.ylabel("Z Score")
            plt.ylim(-5, 5)
            plt.yticks(range(-5, 5, 1))
            plt.legend(loc='best')
            plt.title(title)
            plt.savefig(self.plots_folder + "\\" + title + ".png")
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
            cmap = Experiment.get_cmapi(len(STEREO_BEHAVIORS))
            for i, behavior in enumerate(STEREO_BEHAVIORS):
                behavior_tag = STEREO_DICT[behavior]
                if behavior_tag in {'v', 'j'}:
                    continue
                title_to_save = behavior + " " + length
                if derivative:
                    title_to_save = "Derivative " + title_to_save
                title = title_to_save
                z_score_lst_all_mice_behavior_left = self.photometry_array_left[behavior_tag][length]
                z_score_lst_all_mice_behavior_right = self.photometry_array_right[behavior_tag][length]
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
                             label=behavior + " left")
                    plt.plot(xx[:-1], dxdy_right, "--", color=cmap(i), linewidth=2, alpha=0.9,
                             label=behavior + " right")
                else:
                    plt.plot(xx, mean_all_mice_left, "-", color=cmap(i), linewidth=2, alpha=0.9,
                             label=behavior + " left")
                    plt.plot(xx, mean_all_mice_right, "--", color=cmap(i), linewidth=2, alpha=0.9,
                             label=behavior + " right")

                plt.grid(True, linewidth=0.2, color='grey', linestyle='--', alpha=0.5)
                plt.xticks(range(-window_before, window_after + 1, 2))
                plt.xlabel("Time (sec)")
                plt.ylabel("Z Score")
                plt.ylim(-5, 5)
                plt.legend(loc='best')
                plt.title(title)
                plt.savefig(self.plots_folder + "\\" + title_to_save + ".png")
                plt.show()

    @staticmethod
    def parameterize_z_score(z_score, start_index_in_bout, reduce_from_start, add_to_end, normalize_window_start,
                             normalize_window_end, onset):
        if onset:
            return z_score[start_index_in_bout - reduce_from_start:start_index_in_bout + add_to_end] - np.mean(
                z_score[start_index_in_bout - normalize_window_start:start_index_in_bout - normalize_window_end])
        return z_score[start_index_in_bout - reduce_from_start:start_index_in_bout + add_to_end] - np.mean(
            z_score[start_index_in_bout + normalize_window_start:start_index_in_bout + normalize_window_end])

    @staticmethod
    def get_cmapi(n, name='hsv'):
        '''
        Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.
        '''
        return plt.cm.get_cmap(name, n)

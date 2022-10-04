"""
Citri Lab - Edmond and Lily Safra Brain Research Center - The Hebrew University in Jerusalem ®
This code was written with the help of Itay Shalom and Ben Jerry Gonzales.
"""

from MouseExperiment import *


class ScoreParser:
    """
    Object that hold a mouse z_score relevant data
    """

    @staticmethod
    def is_bout_in_period(tags, action, action_subset, minimal_bout_length):
        """
        Check if in the given range there is a bout in length minimal_bout_left of a specific behavior
        :param tags: range of behaviors prediction
        :param action: the behavior
        :param action_subset: if ==1 then search for any kind of behaviors that action is a subset of
        :param minimal_bout_length: minimal bout length
        :return: True if there is a bout in that period, False other wise.
        """
        counter = 0
        for t in np.arange(len(tags) - 1, -1, -1):
            term_of_search = action in tags[t] if action_subset == 1 else tags[t] == action
            if term_of_search:
                counter += 1
            else:
                if counter > minimal_bout_length:
                    return True
                else:
                    counter = 0
        if counter > minimal_bout_length:
            return True
        else:
            return False

    @staticmethod
    def is_last_seq_in_post_baseline_period(tags, action, action_subset, minimal_bout_length):
        """
        Check if in the given range there is a bout of a specific behavior, where we
        look from the beginning of the asked period
        :param tags: range of behaviors prediction
        :param action: the behavior
        :param action_subset: if ==1 then search for any kind of behaviors that action is a subset of
        :param minimal_bout_length: minimal bout length
        :return: True if there is a bout in that period, False other wise.
        """
        counter = 0
        for t in np.arange(len(tags)):
            term_of_search = action in tags[t] if action_subset == 1 else tags[t] == action
            if term_of_search:
                counter += 1
            else:
                if counter > minimal_bout_length:
                    return True
                else:
                    counter = 0
        if counter > minimal_bout_length:
            return True
        else:
            return False

    @staticmethod
    def find_num_frames_from_last_bout(tags, action, action_subset, minimal_bout_length):
        """
        function that check from a certain time point what is the distance from the last bout
        :param tags: range of behaviors prediction
        :param action: the behavior
        :param action_subset: if ==1 then search for any kind of behaviors that action is a subset of
        :param minimal_bout_length: minimal bout length
        :return: the amount of *frames* that differ between the lsat bout and a certain time point
        """
        counter = 0
        total_amount_of_frames = len(tags)
        for t in np.arange(total_amount_of_frames, -1, -1):
            term_of_search = action in tags[t] if action_subset == 1 else tags[t] == action
            if term_of_search:
                counter += 1
            else:
                if counter > minimal_bout_length:
                    return total_amount_of_frames - (t + counter)
                else:
                    counter = 0
        if counter > minimal_bout_length:
            return total_amount_of_frames - counter
        else:
            return False

    @staticmethod
    def find_distances_from_last_bout(tags, action, action_subset, minimal_time_for_bout=1):
        """
        function that check from a certain time point what is the distance from the last bout
        :param tags: range of behaviors prediction
        :param action: the behavior
        :param action_subset: if ==1 then search for any kind of behaviors that action is a subset of
        :param minimal_time_for_bout: minimal bout length
        :return: the amount of *frames* that differ between the lsat bout and a certain time point
        """
        # todo :  why is that 30 and not 15?
        second = 15  # in Itay code is 30

        window_end_to_bout_onset = 10 * second
        priors_period = 20 * second

        bouts = {'bouts': [], 'time_from_last_bout': []}
        bout = []
        for col in range(priors_period, len(tags) - window_end_to_bout_onset):
            term_of_search = action in tags[col] if action_subset == 1 else tags[col] == action
            if term_of_search:
                bout.append(col)
            else:
                if len(bout) > minimal_time_for_bout * second:
                    time_from_last_bout = ScoreParser.find_num_frames_from_last_bout(
                        tags=tags[bout[0] - priors_period:bout[0]],
                        action=action, action_subset=action_subset, minimal_bout_length=1 * second)
                    if time_from_last_bout:
                        bouts['bouts'].append(bout)
                        bouts['time_from_last_bout'].append(time_from_last_bout)
                bout = []

        return bouts

    @staticmethod
    def find_bouts_long_short_np_wp(tags, action, short_long_threshold, behavior_time_seperation, frames_per_second,
                                    action_subset,
                                    threshold_for_bout_length=1):
        """
        Static method that finds bouts in the given tags_array and return a matrix of the bouts and their tags.
        :param tags: range of behaviors prediction
        :param action: the behavior
        :param short_long_threshold: The threshold that separate between short and long behaviours
        :param behavior_time_seperation: The minmum time between two actions
        :param frames_per_second: Number of frames per second of the z_score
        :param action_subset: if ==1 then search for any kind of behaviors that action is a subset of
        :param threshold_for_bout_length: The threshold that determines the minimum time (in seconds) for a bout.
        """
        second = frames_per_second
        short_long_threshold = short_long_threshold * second
        window_end_to_bout_onset = 10 * second
        priors_period = int(behavior_time_seperation * second)
        np_bouts = {'short': [], 'long': []}
        wp_bouts = {'short': [], 'long': []}
        bout = []
        for col in range(priors_period, len(tags) - window_end_to_bout_onset):
            term_of_search = action in tags[col] if action_subset == 1 else tags[col] == action
            if term_of_search:
                bout.append(col)
            else:
                if len(bout) > threshold_for_bout_length * second:
                    # check in the 10 sec prior if there was no 1 sec bout
                    if not ScoreParser.is_bout_in_period(tags[bout[0] - priors_period:bout[0]], action, action_subset,
                                                         minimal_bout_length=1 * second):
                        # np (Non Prior) - is the situation where we do not have a bout in the behavior_time_seperation
                        # seconds before
                        if len(bout) <= short_long_threshold:
                            np_bouts['short'].append(bout)
                        else:
                            np_bouts['long'].append(bout)
                    else:
                        # wp (With Prior) - is the situation where we do have a bout in the 10 sec before
                        if len(bout) <= short_long_threshold:
                            wp_bouts['short'].append(bout)
                        else:
                            wp_bouts['long'].append(bout)

                bout = []

        return np_bouts, wp_bouts

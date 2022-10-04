"""
This file was created by Niv Ben Salmon,
Citri Lab - Edmond and Lily Safra Brain Research Center - The Hebrew University in Jerusalem ®
This code was written with the help of Itay Shalom and Ben Jerry Gonzales.
"""
from MouseExperiment import *
from Experiment import Experiment


def offset_analysis():
    '''
    Function that process the experiment folder according to Offset bouts. Plots the requested grapch according
    to the parameters inside the function
    '''
    # # # *** Analyzing both hemispheres offset and plotting them together ***
    photometry_bouts_offset = experiment.analyze_photometry_traces_bout_both_hemispheres(z_score_window_start=5,
                                                                                         z_score_window_end=5,
                                                                                         normalize_window_start=2,
                                                                                         normalize_window_end=5,
                                                                                         behavior_time_seperation=5,
                                                                                         onset=False)
    experiment.plot_signals_according_to_old_code(photometry_bouts_offset, window_before=5, window_after=5,
                                                  plot_mouse_cases=True,
                                                  given_heading="offset")

    # # # *** Analyzing each hemisphere differently offset ***
    photometry_bouts_offset_left, photometry_bouts_offset_right = \
        experiment.analyze_photometry_traces_bout_different_hemispheres(z_score_window_start=5, z_score_window_end=5,
                                                                        normalize_window_start=2,
                                                                        normalize_window_end=5,
                                                                        behavior_time_seperation=10,
                                                                        onset=False)
    experiment.plot_signals_according_to_old_code(photometry_bouts_offset_right, window_before=5, window_after=5,
                                                  plot_mouse_cases=True,
                                                  given_heading="offset left")
    experiment.plot_signals_according_to_old_code(photometry_bouts_offset_left, window_before=5, window_after=5,
                                                  plot_mouse_cases=True,
                                                  given_heading="offset right")

    # plot all action together - if normal z_score and not derivative, just set derivative=False
    experiment.plot_actions_separately_for_each_hemisphere(derivative=True, window_before=5, window_after=5)
    experiment.plot_actions_together_for_each_hemisphere(derivative=True, window_before=5, window_after=5)

    experiment.plot_actions_separately_for_each_hemisphere(derivative=False, window_before=5, window_after=5)
    experiment.plot_actions_together_for_each_hemisphere(derivative=False, window_before=5, window_after=5)

    # plot all action together - normalize according to the max of behavior/max of all actions
    experiment.plot_actions_together_for_each_hemisphere_normalize_according_to_max_min(max_val=1, min_val=0,
                                                                                        window_before=5,
                                                                                        window_after=5)


def onset_analysis():
    # # *** Analyzing both hemispheres onset and plotting them together ***
    photometry_bouts_onset = experiment.analyze_photometry_traces_bout_both_hemispheres(z_score_window_start=10,
                                                                                        z_score_window_end=20,
                                                                                        normalize_window_start=10,
                                                                                        normalize_window_end=4,
                                                                                        behavior_time_seperation=10,
                                                                                        onset=True)
    experiment.plot_signals_according_to_old_code(photometry_bouts_onset, window_before=10, window_after=20,
                                                  plot_mouse_cases=True,
                                                  given_heading="onset")

    # *** Analyzing each hemisphere onset differently ***
    photometry_bouts_onset_left, photometry_bouts_onset_right = \
        experiment.analyze_photometry_traces_bout_different_hemispheres(z_score_window_start=10, z_score_window_end=20,
                                                                        normalize_window_start=10,
                                                                        normalize_window_end=4,
                                                                        behavior_time_seperation=10,
                                                                        onset=True)
    experiment.plot_signals_according_to_old_code(photometry_bouts_onset_left, window_before=10, window_after=20,
                                                  plot_mouse_cases=True,
                                                  given_heading="onset left")
    experiment.plot_signals_according_to_old_code(photometry_bouts_onset_right, window_before=10, window_after=20,
                                                  plot_mouse_cases=True,
                                                  given_heading="onset right")

    # # plot all action together - if normal z_score and not derivative, just set derivative=False
    experiment.plot_actions_separately_for_each_hemisphere(derivative=True)
    experiment.plot_actions_together_for_each_hemisphere(derivative=True)

    experiment.plot_actions_separately_for_each_hemisphere(derivative=False)
    experiment.plot_actions_together_for_each_hemisphere(derivative=False)

    # plot all action together - normalize according to the max of behavior/max of all actions
    experiment.plot_actions_together_for_each_hemisphere_normalize_according_to_max_min(max_val=1, min_val=0)


def process_experiment():
    exp = Experiment(exp_folder, manual_tag=manual_tagged)
    for mice_folder in os.listdir(exp_folder):
        if 'plots' in mice_folder or 'key' in mice_folder:
            continue
        mouse = Mouse(os.path.join(exp.path_to_experiment, mice_folder), exp)
        exp.add_mouse(mouse)

    return exp


if __name__ == "__main__":
    '''
    Pick a folder to process
    '''
    exp_folder = r"path to experiment"
    ''''
    manual_tagged == -1: tagged manualy - Find all shows that contains act letters, == 0 tagged by Stereo,
    == 1 tagged manualy - find only specific shows of this act
    '''
    manual_tagged = 0
    experiment = process_experiment()
    onset_analysis()
    offset_analysis()

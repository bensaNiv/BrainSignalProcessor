"""
This file was created by Niv Ben Salmon,
Citri Lab - Edmond and Lily Safra Brain Research Center - The Hebrew University in Jerusalem ®
This code was written with the help of Itay Shalom and Ben Jerry Gonzales.
"""
from MouseExperiment import *
from Experiment import Experiment


if __name__ == "__main__":
    exp_folder = r"..."  # add you folder to provess here
    experiment = Experiment(exp_folder)
    for mice_folder in os.listdir(exp_folder):
        if 'plots' in mice_folder:
            continue
        mouse = Mouse(os.path.join(experiment.path_to_experiment, mice_folder), experiment)
        experiment.add_mouse(mouse)

    # # *** Analyzing both hemispheres onset and plotting them together ***
    photometry_bouts_onset = experiment.analyze_photometry_traces_bout_both_hemispheres(10, 20, 10, 4)
    experiment.plot_signals_according_to_old_code(photometry_bouts_onset, window_before=10, window_after=20,
                                                  plot_mouse_cases=True,
                                                  given_heading="onset")

    # *** Analyzing each hemisphere onset differently ***
    photometry_bouts_onset_left, photometry_bouts_onset_right = \
        experiment.analyze_photometry_traces_bout_different_hemispheres(10, 20, 10, 4)
    experiment.plot_signals_according_to_old_code(photometry_bouts_onset_left, window_before=10, window_after=20,
                                                  plot_mouse_cases=True,
                                                  given_heading="onset left")
    experiment.plot_signals_according_to_old_code(photometry_bouts_onset_right, window_before=10, window_after=20,
                                                  plot_mouse_cases=True,
                                                  given_heading="onset right")

    # plot all action together - if normal z_score and not derivative, just set derivative=False
    experiment.plot_actions_separately_for_each_hemisphere(derivative=True)
    experiment.plot_actions_together_for_each_hemisphere(derivative=True)

    experiment.plot_actions_separately_for_each_hemisphere(derivative=False)
    experiment.plot_actions_together_for_each_hemisphere(derivative=False)

    # plot all action together - normalize according to the max of behavior/max of all actions
    experiment.plot_actions_together_for_each_hemisphere_normalize_according_to_max()
    
    # #
    # # # *** Analyzing both hemispheres offset and plotting them together ***
    photometry_bouts_offset = experiment.analyze_photometry_traces_bout_both_hemispheres(5, 5, 2, 5, onset=False)
    experiment.plot_signals_according_to_old_code(photometry_bouts_offset, window_before=5, window_after=5,
                                                  plot_mouse_cases=True,
                                                  given_heading="offset")

    # # *** Analyzing each hemisphere differently offset ***
    photometry_bouts_offset_left, photometry_bouts_offset_right = \
        experiment.analyze_photometry_traces_bout_different_hemispheres(5, 5, 2, 5, onset=False)
    experiment.plot_signals_according_to_old_code(photometry_bouts_offset_right, window_before=5, window_after=5,
                                                  plot_mouse_cases=True,
                                                  given_heading="offset left")
    experiment.plot_signals_according_to_old_code(photometry_bouts_offset_left, window_before=5, window_after=5,
                                                  plot_mouse_cases=True,
                                                  given_heading="offset right")

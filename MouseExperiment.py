import datetime
from Signal import *
import os
import pickle

STEREO_DICT = {'Grooming': 'g', 'Body licking': 'ld', 'Wall licking': 'lw', 'Floor licking': 'lf',
               'Rearing': 'r', 'Back to camera': 'v', 'Other': 'n', 'Immobile': 'i', 'Jump': 'j'}

STEREO_DICT_REV = {'g': 'Grooming', 'ld': 'Body licking', 'lw': 'Wall licking', 'lf': 'Floor licking',
                   'r': 'Rearing', 'v': 'Back to camera', 'n': 'Other', 'i': 'Immobile', 'j': 'Jump'}

STEREO_BEHAVIORS = ['Grooming', 'Body licking', 'Wall licking', 'Floor licking', 'Rearing', 'Back to camera', 'Other',
                    'Immobile', 'Jump']

STEREO_COLOR_DICT = {'Grooming': '#c21296', 'Body licking': '#06d6a0', 'Wall licking': '#ee476f',
                     'Floor licking': '#1189b1', 'Rearing': '#ffd169',
                     'Back to camera': '#b2b2b2', 'Other': '#dadada', 'Immobile': '#f6f6f6', 'Jump': '#2A363B'}

ACTION_DICT = {'n': ('All', 'Other/Move'),
               'v': ('All', 'Not Visible'),
               'i': ('All', 'Immobile'),
               'j': ('All', 'Jump'),
               'b': ('Oral', 'Bite'),
               'l': ('Oral', 'Lick'),
               'u': ('Upper Limbs', 'Push/pull (Obj/HandsMo)'),
               'g': ('Upper Limbs', 'Swipe (Groom)'),
               'x': ('Upper Limbs', 'Biceps-Flexed'),
               'z': ('Rear-paw', 'Groom'),
               'r': ('Rear-paw', 'Rearing'),
               'c': ('Obj', 'Central/cotton'),
               'w': ('Obj', 'Wall'),
               'f': ('Obj', 'Floor'),
               'a': ('Obj', 'Head'),
               'd': ('Obj', 'Body'),
               'h': ('Obj', 'Hands/upper-limbs'),
               'y': ('Obj', 'Rear-paws/tail'),
               'p': ('Obj', 'Poop')}


class Mouse:
    """
    Object that represent a mouse in a specific experiment
    """

    def __init__(self, path_file: str, experiment):
        self.path_to_mouse_folder = path_file
        self.experiment = experiment
        self.amount_of_tags_in_second = 0
        self.name = self.parse_mouse_name()
        self.tags_array = self.create_tags_array()
        self.left_signal, self.right_signal = self.create_signal_for_mouse()
        self.tags_frame_rate = self.right_signal.frame_rate
        self.adapt_array()

    def create_signal_for_mouse(self):
        poly_deg = 3
        baseline_window_in_minute = 2
        signal_iso, s_470_path = '', ''
        signals_region = {'l': "Region2G", 'r': "Region3G"}
        signals_obj = {'l': None, 'r': None}
        iso_path, sig470_path = "", ""
        pattern_iso = re.compile(r'(415)')
        pattern_470 = re.compile(r'(470)')
        results_dir = os.path.join(self.path_to_mouse_folder,
                                   'Results Signal' + str(datetime.datetime.now()).replace(":", "-")[:-10])
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        for file_name in os.listdir(self.path_to_mouse_folder):
            if re.search(pattern_iso, file_name):
                iso_path = file_name
                continue
            if re.search(pattern_470, file_name):
                s_470_path = file_name
                continue
        for side in signals_region.keys():
            signals_obj[side] = Signal(self.path_to_mouse_folder, "\\" + s_470_path, signals_region[side],
                                       "\\" + iso_path, results_dir, len(self.tags_array),
                                       self.amount_of_tags_in_second, poly_deg,
                                       baseline_window_in_minute)

        return signals_obj['l'], signals_obj['r']

    def create_tags_array(self):
        tags_folder = os.path.join(self.path_to_mouse_folder, 'tags')
        if not os.path.exists(tags_folder):
            print("No tags folder for mouse: " + self.path_to_mouse_folder.split("\\")[-1])
            return None
        i = 0
        df_tags = []
        for path in os.listdir(tags_folder):
            if i > 0:
                assert "More then one file in the tags directory"
            i += 1
            tags_file = os.path.join(tags_folder, path)
            # tags array was tagged manually
            if self.experiment.manual_tag:
                self.amount_of_tags_in_second = 30
                try:
                    df = pd.read_excel(tags_file)
                    df_tags = np.array(df).reshape(df.shape[0])
                except Exception:
                    try:
                        df = pd.read_csv(tags_file)
                        df_tags = np.array(df).reshape(df.shape[0])
                    except Exception:
                        assert "Error in creatig the tags array"
                behavior_set = set(df_tags)
                self.experiment.behaviors_container |= behavior_set
            # tags array came from automatic tagger
            if os.path.splitext(tags_file)[1] == ".pkl":
                if len(self.experiment.behaviors_container) == 0:
                    self.experiment.behaviors_container = list(STEREO_DICT.values())
                self.amount_of_tags_in_second = 15
                with open(tags_file, 'rb') as f:
                    data = pickle.load(f)
                predictions = list((data['merged']['predictions']).astype(int))
                df_tags = list(map(lambda l: STEREO_DICT[STEREO_BEHAVIORS[l]], predictions))
        return df_tags

    def adapt_array(self):
        """
        Function that adapt the tags array according to signal frame rate
        :return:
        """
        cut = 30
        # In case where the frame rate are not equal
        if self.amount_of_tags_in_second != self.tags_frame_rate:
            common_divider = Signal.gcd(self.amount_of_tags_in_second, self.tags_frame_rate)
            jump = self.amount_of_tags_in_second // common_divider
            num_frames_in_array = self.right_signal.num_frames_in_array
            n = num_frames_in_array * jump
            self.tags_array = self.tags_array[:n]
            self.tags_array = self.tags_array[::jump]
            # n = len(self.tags_array)//jump * jump
            cut *= common_divider
            n = len(self.tags_array)
        else:
            cut *= self.amount_of_tags_in_second
            n = min(len(self.tags_array), len(self.left_signal.z_score)) + cut
            # self.tags_array = self.tags_array[:n]

        # update the amout of tags per frame in the expirement
        if self.experiment.frames_per_second_expiriment == self.experiment.FRAMES_NOT_SET:
            self.experiment.set_frames_in_expiriment(self.tags_frame_rate)

        self.tags_array = self.tags_array[cut:n]

    def parse_mouse_name(self):
        """
        Parse the mouse name from the path of the mouse folder
        :return: The mouse name in the form : CGxxxMSxxx
        """
        dir_name = os.path.split(self.path_to_mouse_folder)[1]
        return dir_name[dir_name.find("CG"):].split("_")[0]

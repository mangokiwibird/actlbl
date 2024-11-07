import numpy as np

import settings

FRAMES_PER_SAMPLE = settings.get_frames_per_sample()
MAX_FRAMES_IN_HISTORY = settings.get_max_frames_in_history()


class FrameLoader:
    """Frame Loader class

    This class saves a sequence of frames into the 'history' attribute, and ultimately saves the data into a json file

    Attributes:
        data_target: The target JSON filename
        history: Sequence of frames. Stacks up to 100, and discards the oldest keypoint after that.

    """

    def __init__(self):
        """Initializer

        Args:
            data_target: The target JSON filename
        """

        self.history = []

    def save_frame(self, keypoints):
        if len(self.history) == MAX_FRAMES_IN_HISTORY:
            self.history.pop(0)  # pop the first element once the limit condition is met

        self.history.append(keypoints)

    def get_data(self):
        return np.array(self.history).tolist()

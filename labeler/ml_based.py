#     Copyright (C) 2024 dolphin2410
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

import json

import numpy as np
import tensorflow as tf
import settings
from labeler.common import Labeler
from labeler.ml_model import train_labeler, preprocess_data

FRAMES_PER_SAMPLE = settings.get_frames_per_sample()


class MLBasedLabeler(Labeler):
    """
    Labels activity using keypoints parsed by movenet.

    Attributes:
        history:
            A list of past keypoints. Stacks up to 100, and discards the oldest keypoint after that.
    """

    history = []

    def __init__(self, data_target: str, model_path: str = None):
        """
        Initializer

        Args:
            data_target: filename target to save json data
        """

        self.data_target = data_target

        # if model_path is not None:
        #     self.model = tf.keras.models.load_model(model_path)
        # else:
        #     self.model = train_labeler()

    # TODO: implement
    def get_score(self, keypoints):
        """Returns dictionary of [action - score]"""

        flattened_train_data = np.zeros((1, 34 * FRAMES_PER_SAMPLE, 1))  # TODO: remove hardcoding
        if len(self.history) <= FRAMES_PER_SAMPLE:
            return {"NO_DATA": 1}

        flattened_train_data[0, :, 0] = preprocess_data(np.array(self.history)[-FRAMES_PER_SAMPLE:]).flatten()
        self.model.predict(flattened_train_data, verbose=0)

        return {"NO_DATA": 1}

    def save_frame(self, keypoints):
        """
        Saves keypoints to history

        Args:
            keypoints:
                Keypoints parsed by movenet
        """

        if len(self.history) == 100:  # TODO: remove hardcoding
            self.history.pop(0)

        self.history.append(keypoints)

    def save_data(self):
        """Saves history data to json file"""

        with open(self.data_target, "w") as outfile:
            json.dump({"history": np.array(self.history[-FRAMES_PER_SAMPLE:]).tolist()}, outfile)
            print("successfully saved data to file")

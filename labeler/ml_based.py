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

from labeler.ml_model import preprocess_data, train_labeler
import tensorflow as tf


class MLBasedLabeler:
    """
    Labels activity using keypoints parsed by movenet.

    Attributes:
        history:
            A list of past keypoints. Stacks up to 100, and discards the oldest keypoint after that.
    """

    history = []

    def __init__(self, data_target: str):
        # self.model = tf.keras.models.load_model('model/hangang_cat.keras')
        # self.model = train_labeler()
        self.data_target = data_target

    def append_keypoints_to_history(self, keypoints):
        """
        Saves keypoints to history

        Args:
            keypoints:
                Keypoints parsed by movenet
        """

        if len(self.history) == 25:
            self.history.pop(0)

        self.history.append(np.array(keypoints).tolist()[0][0])

    # TODO: implement
    def get_score(self, keypoints):
        """Returns dictionary of [action - score]"""

        self.append_keypoints_to_history(keypoints)
        # flattened_train_data = np.zeros((1, 22 * 25, 1))  # TODO: remove hardcoding
        # if len(self.history) >= 25:
        #     flattened_train_data[0, :, 0] = preprocess_data(np.array(self.history)[:25]).flatten()
        #     predicted_data = self.model.predict(flattened_train_data, verbose=0)
        #     percentage = predicted_data[0] / np.sum(predicted_data[0])
        #     return { "꽁꽁": percentage[0], "얼어붙은": percentage[1], "한강위로": percentage[2], "고양이가": percentage[3], "걸어다닙니다": percentage[4] }
        # else:
        return {"NO_DATA" : 1}

    def save_data(self):
        """Saves history data to json file"""

        with open(self.data_target, "w") as outfile:
            json.dump({"history": self.history}, outfile)
            print("successfully saved data to file")

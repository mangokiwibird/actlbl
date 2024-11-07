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

import numpy as np
import tensorflow as tf
import settings
from labeler.common import Labeler

FRAMES_PER_SAMPLE = settings.get_frames_per_sample()


class MLBasedLabeler(Labeler):
    """Labels activity using keypoints parsed by movenet."""

    def __init__(self, model_path: str = None):
        """Initializer

        Args:
            data_target: filename target to save json data
            model_path: path of keras model
            classified_history: Data to feed
        """

        super().__init__()

        if model_path is not None:
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = None

    def get_score(self):
        if self.model is None:
            raise Exception("Invalid Model. No data fed.")

        if len(self.history) < FRAMES_PER_SAMPLE:
            return {"PREDICTION": -1, "PROBABILITY": []}
        
        predict_result = self.model.predict(np.array([self.history]), verbose=0)[0]

        return {"PREDICTION": predict_result.index(max(predict_result)), "PROBABILITY": predict_result}

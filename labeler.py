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


class MLBasedLabeler:
    """
    Labels activity using keypoints parsed by movenet.

    Attributes:
        history:
            A list of past keypoints. Stacks up to 100, and discards the oldest keypoint after that.
    """

    history = []

    def append_keypoints_to_history(self, keypoints):
        """
        Saves keypoints to history

        Args:
            keypoints:
                Keypoints parsed by movenet
        """

        if len(self.history) == 100:
            self.history.pop(0)

        self.history.append(np.array(keypoints).tolist()[0][0])

    # TODO: implement
    def get_score(self, keypoints):
        """Returns dictionary of [action - score]"""

        self.append_keypoints_to_history(keypoints)

        return {"walking": 1.0}

    def save_data(self):
        """Saves history data to json file"""

        with open("./data.json", "w") as outfile:
            json.dump({"history": self.history}, outfile)
            print("successfully saved data to file")

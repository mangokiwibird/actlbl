#     Copyright (C) 2024 gravitaionalacceleration
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

from labeler.common import Labeler
from movenet import KEYPOINT_LEFT_HIP, KEYPOINT_RIGHT_HIP, KEYPOINT_LEFT_ANKLE, \
    KEYPOINT_RIGHT_ANKLE, KEYPOINT_LEFT_SHOULDER, KEYPOINT_RIGHT_SHOULDER


class RuleBasedLabeler(Labeler):

    history = []
    probability_history = []

    previous_keypoints = None
    prevact = None
    activity_statistics = [np.zeros(5) for _ in range(5)]

    def get_speed(self, current_keypoints, direction='x'):
        if self.previous_keypoints is None:
            return 0

        if direction == 'x':
            previous_keypoints = self.previous_keypoints[:, 0]
            current_keypoints = current_keypoints[:, 1]
        else:
            previous_keypoints = self.previous_keypoints[:, 0]
            current_keypoints = current_keypoints[:, 1]

        ignore_indices = [0, 1, 2, 3, 4, 7, 8, 9, 10]

        mask = np.ones(previous_keypoints.shape[0], dtype=bool)
        mask[ignore_indices] = False

        previous_keypoints = previous_keypoints[mask]
        current_keypoints = current_keypoints[mask]

        displacement = np.sqrt(np.sum((current_keypoints - previous_keypoints) ** 2))
        mean_displacement = np.mean(displacement)

        return mean_displacement

    def append_keypoints_to_history(self, keypoints):
        if len(self.history) == 25:
            self.history.pop(0)
        self.history.append(np.array(keypoints).tolist()[0][0])

    def update_statistics(self, activity):
        activity_list = ["lying", "sitting", "standing", "walking", "running"]

        print(self.activity_statistics)
        print(activity)

        # if self.prevact in activity_list and activity in self.activity_statistics:
        #     self.activity_statistics[activity_list.index(self.prevact)][activity_list.index(activity)] += 1

    def activity(self, keypoints, prevact):
        excluded_indices = [0, 1, 2, 3, 4, 7, 8, 9, 10]
        filtered_keypoints = np.delete(keypoints, excluded_indices, axis=0)

        x_points, y_points = filtered_keypoints[:, 0], filtered_keypoints[:, 1]

        x_min, x_max = x_points.min(), x_points.max()
        y_min, y_max = y_points.min(), y_points.max()
        width = x_max - x_min
        height = y_max - y_min

        ratio = width / height

        lying_ratio_threshold = 1.0

        threshold_height = 0.42
        speed = self.get_speed(keypoints)

        hip_center = (keypoints[KEYPOINT_LEFT_HIP] + keypoints[KEYPOINT_RIGHT_HIP]) / 2
        ankle_center = (keypoints[KEYPOINT_LEFT_ANKLE] + keypoints[KEYPOINT_RIGHT_ANKLE]) / 2

        # center = ankle_center - hip_center
        shoulder_center = (keypoints[KEYPOINT_LEFT_SHOULDER] + keypoints[KEYPOINT_RIGHT_SHOULDER]) / 2
        height_center = ankle_center - shoulder_center
        center = height_center / hip_center

        print(speed)

        if ratio < lying_ratio_threshold:
            return "lying"
        if prevact == "lying":
            if ratio < lying_ratio_threshold:
                return "lying"
            return "sitting"
        if prevact == "sitting":
            if center[0] < threshold_height:
                return "sitting"
            return "standing"
        if prevact == "standing":
            if center[0] < threshold_height:
                return "sitting"
            if speed > 0.013:
                return "walking"
            return "standing"
        if prevact == "walking":
            if speed > 0.05:
                return "running"
            if speed > 0.013:
                return "walking"
            return "standing"
        return "really nothing???"

    def save_frame(self, keypoints):
        pass

    def get_score(self, keypoints):
        self.append_keypoints_to_history(keypoints)

        activity = self.activity(keypoints, self.prevact)

        self.update_statistics(activity)

        # print(self.activity_statistics)

        self.prevact = activity
        self.previous_keypoints = keypoints

        # note that activity replaces one of the five declared previously
        return {"walking": 0, "running": 0, "standing": 0, "sitting": 0, "lying": 0, activity: 1}

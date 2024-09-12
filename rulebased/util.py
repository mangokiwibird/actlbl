import math

import numpy as np

from movenet import KeypointGroup, Keypoint


def calculate_angle_coords(p1, p2, p3):
    v1 = np.abs(p1 - p2)  # vector 1: points from p1 to p2
    v2 = np.abs(p2 - p3)  # vector 1: points from p1 to p2
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cosine_angle)
    if angle is None:
        return 0
    return np.degrees(angle)


def calculate_angle(keypoints, keypoint1: Keypoint, keypoint2: Keypoint, keypoint3: Keypoint):
    return calculate_angle_coords(keypoints[keypoint1.value], keypoints[keypoint2.value], keypoints[keypoint3.value])


def calculate_average_velocity(previous_keypoints, current_keypoints):
    if previous_keypoints is None:
        return 0

    mask = np.ones(previous_keypoints.shape[0], dtype=bool)

    previous_keypoints = previous_keypoints[mask]
    current_keypoints = current_keypoints[mask]

    displacement = current_keypoints - previous_keypoints

    displacement_x = np.mean(displacement[:, 1])
    displacement_y = np.mean(displacement[:, 0])

    return displacement_x, displacement_y


def calculate_average_speed(previous_keypoints, current_keypoints):
    """Calculates displacement between ticks"""

    vx, vy = calculate_average_velocity(previous_keypoints, current_keypoints)

    return math.sqrt(vx ** 2 + vy ** 2)


def get_center_coords(keypoints, keypoint_group: KeypointGroup):
    if keypoint_group.value == 0:
        return keypoints[0][0:2]
    return (keypoints[keypoint_group.value * 2 - 1][0:2] + keypoints[keypoint_group.value * 2][0:2]) / 2


def calculate_x_distance(keypoints, keypoint_group1: KeypointGroup, keypoint_group2: KeypointGroup):
    group1_center = get_center_coords(keypoints, keypoint_group1)
    group2_center = get_center_coords(keypoints, keypoint_group2)

    x_distance = math.fabs(group1_center[0] - group2_center[0])
    return x_distance


def calculate_y_distance(keypoints, keypoint_group1: KeypointGroup, keypoint_group2: KeypointGroup):
    group1_center = get_center_coords(keypoints, keypoint_group1)
    group2_center = get_center_coords(keypoints, keypoint_group2)

    y_distance = math.fabs(group1_center[1] - group2_center[1])
    return y_distance


def calculate_distance(keypoints, keypoint_group1: KeypointGroup, keypoint_group2: KeypointGroup):
    group1_center = get_center_coords(keypoints, keypoint_group1)
    group2_center = get_center_coords(keypoints, keypoint_group2)

    distance = math.sqrt((group1_center[0] - group2_center[0]) ** 2 + (group1_center[1] - group2_center[1]) ** 2)
    return distance


class RBContext:
    def __init__(self):
        self.settings = {}

# activity
import vectors

import numpy as np

from movenet import KeypointGroup, Keypoint
from rulebased import util
from rulebased.util import calculate_angle


def sitting_frontcamera(keypoints):
    front_height = util.calculate_distance(keypoints, KeypointGroup.EYE, KeypointGroup.ANKLE)
    return front_height


def sitting_sidecamera(keypoints):
    """1. knee와 hib사이의 거리를 x축(length_x)에서, y축(length_y)에서로 나눔 length_x>length_y 이면 앉을 상태 length_x<length_y면 일어선 상태일 것임
       2. shoulder, hib, knee 사이의 각도를 얻어내고 그 각도가 어떤 임계값들에 따라 lying, sitting, standing으로 나누어짐"""

    length_x = util.calculate_x_distance(keypoints, KeypointGroup.HIP, KeypointGroup.KNEE)
    length_y = util.calculate_y_distance(keypoints, KeypointGroup.HIP, KeypointGroup.KNEE)

    """help--
    Distance between two wrists, two knees, and two ankle will constant when object is sitting
    Use this code when sitting confidence is low
    accuracy[0]:wrist, accuracy[1]:knee, accuracy[2]: ankle
    accuracy = [np.abs(keypoints[10]-keypoints[9])]
    for i in range(13, 15, 2):
        accuracy += np.abs(keypoints[i+1]-keypoints[i])"""

    angles = [
        calculate_angle(keypoints, Keypoint.LEFT_SHOULDER, Keypoint.LEFT_HIP, Keypoint.LEFT_KNEE),
        calculate_angle(keypoints, Keypoint.RIGHT_SHOULDER, Keypoint.RIGHT_HIP, Keypoint.RIGHT_KNEE),
    ]

    return length_x, length_y, angles

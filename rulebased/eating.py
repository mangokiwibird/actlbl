import math

import numpy as np

from movenet import Keypoint, KeypointGroup
from rulebased.util import calculate_speed, RBContext, calculate_distance


def eating(ctx: RBContext, keypoints):
    lelbow = keypoints[Keypoint.LEFT_ELBOW]
    relbow = keypoints[Keypoint.RIGHT_ELBOW]
    lwrist = keypoints[Keypoint.LEFT_WRIST]
    rwrist = keypoints[Keypoint.RIGHT_WRIST]

    velocity_lwrist = calculate_speed(ctx.settings.get("PREV_POS_LWRIST"), lwrist)
    velocity_rwrist = calculate_speed(ctx.settings.get("PREV_POS_RWRIST"), rwrist)

    ctx.settings["PREV_POS_LWRIST"] = lwrist
    ctx.settings["PREV_POS_RWRIST"] = rwrist

    state = 1

    if velocity_lwrist >= 0 or velocity_rwrist >= 0:
        if velocity_lwrist > velocity_rwrist:
            x_length = math.fabs(lelbow[1] - relbow[1])
            y_length = math.fabs(lelbow[0] - relbow[0])
            elbow_length = (x_length ** 2 + y_length ** 2) ** 0.5

            if ctx.settings.get("PREV_LENGTH", 0) - elbow_length < 0:
                state = 2
            ctx.settings["PREV_LENGTH"] = elbow_length
            return state, velocity_lwrist, velocity_rwrist
        elif velocity_lwrist < velocity_rwrist:
            elbow_wrist_length = calculate_distance(keypoints, KeypointGroup.ELBOW, KeypointGroup.WRIST)

            if ctx.settings.get("PREV_LENGTH", 0) < elbow_wrist_length:
                state = 2

            ctx.settings["PREV_LENGTH"] = elbow_wrist_length
            return state, velocity_lwrist, velocity_rwrist

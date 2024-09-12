from movenet import KeypointGroup
from rulebased.util import get_center_coords, calculate_angle, calculate_angle_coords


def is_raising_hand(keypoints):
    shoulder_center = get_center_coords(keypoints, KeypointGroup.SHOULDER)
    elbow_center = get_center_coords(keypoints, KeypointGroup.ELBOW)
    wrist_center = get_center_coords(keypoints, KeypointGroup.WRIST)

    angle = calculate_angle_coords(shoulder_center, elbow_center, wrist_center)
    threshold_angle = 180  # TODO: 임의로 정한 것
    if angle < threshold_angle and elbow_center[0] > shoulder_center[0]:  # TODO 이거 아닌데....
        return True

    return False

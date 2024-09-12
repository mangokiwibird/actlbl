from movenet import KeypointGroup
from rulebased import util
from rulebased.util import get_center_coords, RBContext

JUMP_THRESHOLD = 1.0


def is_jumping(context: RBContext, keypoints):
    # TODO: PREV_KEYPOINTS Must be Set
    vx, vy = util.calculate_average_velocity(context.settings.get("PREV_KEYPOINTS", None), keypoints)
    if vy > JUMP_THRESHOLD:
        return True

    return False

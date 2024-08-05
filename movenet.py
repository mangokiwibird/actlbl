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

from settings import get_model_path

KEYPOINT_INDEX_TO_NAME = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]

KEYPOINT_NAME_TO_INDEX = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}


def filter_not(list_keypoints):
    """Returns an index list of keypoints that weren't passed to list_keypoints

    Args:
        list_keypoints: List of keypoint names

    Returns:
        Index list of keypoints that isn't contained in list_keypoints
    """

    index_list = np.array(range(17))
    index_list = np.delete(
        index_list,
        list(map(lambda keypoint_name: KEYPOINT_NAME_TO_INDEX[keypoint_name], list_keypoints)))

    return index_list


def load_model():
    """Loads Movenet Model"""

    global interpreter

    interpreter = tf.lite.Interpreter(model_path=get_model_path())
    interpreter.allocate_tensors()


def parse_keypoints(image: np.ndarray):
    """Returns a list of keypoints parsed from the given image.

    Movenet model should be loaded before call of this function

    Args:
        image: Image to parse keypoints from

    Returns:
        A list of raw keypoints data, referred as "frame" in this project
    """

    # preprocess image before passing it to movenet
    image = np.expand_dims(image, axis=0)
        
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])
    if keypoints.size == 0:
        raise ValueError("No keypoints detected")

    return keypoints

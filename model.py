import numpy as np
import tensorflow as tf

KEYPOINT_INDEX = [
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

def filter_not(list_keypoints):
    index = np.array(range(17))
    index = np.delete(index, list(map(lambda x: KEYPOINT_DICT[x], list_keypoints)))
    return index

KEYPOINT_DICT = {
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

def load_model():
    """
    Loads Movenet
    """

    global interpreter

    interpreter = tf.lite.Interpreter(model_path="./model/movenet_thunder.tflite")
    interpreter.allocate_tensors()
    return interpreter

def parse_keypoints(image):
    """
    Returns a list of keypoints parsed from the given image. Movenet model should be loaded before call of this function
    """
    image = np.expand_dims(image, axis=0)
        
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])
    if keypoints.size == 0:
        raise ValueError("No keypoints detected")
    return keypoints
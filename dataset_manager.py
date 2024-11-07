import json
import os

import numpy as np

import settings
from camera import start_capture
from image import LabeledImage
from keras.src.utils import pad_sequences


def parse_video_path(video_file: str):
    """Splits video path: action_index.mp4 -> [action, index]"""

    raw_split_video_path = video_file.split(".mp4")[0].split("_")

    # validate file name else return None
    if len(raw_split_video_path) != 2 or not raw_split_video_path[1].isdigit():
        return None

    activity_name = raw_split_video_path[0]

    return activity_name


def load_class(class_directory: str):
    """Loads JSON dataset from given directory

    Converts JSON file into python dictionary.

    Args:
        directory: Name of the directory containing json files with history data
    Returns:
        Dictionary with history data
    """

    history_list = []

    history_files = os.listdir(class_directory)
    for history_path in history_files:
        history_full_path = os.path.join(class_directory, history_path)
        
        with open(history_full_path, "r") as json_file:
            json_data = json.load(json_file)
            history = json_data["history"]
            history_list.append(np.array(history))

    return history_list

def load_history_dataset(dataset: str):
    """
    Loads the given dataset directory. There should be a subdirectory for each action class inside the dataset folder.
    """

    train_data = []
    labels = []

    files = os.listdir(dataset)

    for (class_index, path) in enumerate(files):
        canonical_path = os.path.join(dataset, path)
        if not os.path.isdir(canonical_path):
            continue
        
        history_class = load_class(canonical_path)

        for history in history_class:
            train_data.append(history)
            labels.append(class_index)

    return train_data, labels

def augment_dataset(history_list, labels, N=30):
    new_history_list = []
    new_labels = []
    for history_index, history in enumerate(history_list):
        for i in range(N):
            new_history = []
            for scene in history:
                new_scene = []
                for keypoint in scene:
                    x, y, confidence = keypoint
                    new_keypoint = np.array([x, y, confidence])
                    new_scene.append(new_keypoint)
                new_history.append(new_scene)
            new_history_list.append(new_history)
            new_labels.append(labels[history_index])
    return new_history_list, new_labels

def history_add_padding(history_list, max_shift=0.01):
    N = 100000000.0

    new_history_list = []
    for history in history_list:
        new_history = []
        for scene in history:
            new_scene = []
            for keypoint in scene:
                x, y, confidence = keypoint
                new_keypoint = np.array([x * N, y * N, confidence * N])
                new_keypoint += np.random.uniform(0, max_shift, 3)
                new_scene.append(new_keypoint)
            new_history.append(new_scene)
        new_history_list.append(new_history)

    padded_inputs = pad_sequences(new_history_list, padding="post") / N

    return padded_inputs

def extract_keypoints(video_path):
    def callback(img: LabeledImage):
        img.record_activity()
        if "counter" in img.camera_context.settings:
            img.camera_context.settings["counter"] += 1
        else:
            img.camera_context.settings["counter"] = 1

    ctx = start_capture(video_path, callback)

    if settings.is_debug():
        current_frames = int(ctx.settings.get("counter"))
        print(f"Frames in current video: {current_frames}")

    return ctx.ml_labeler.get_data()


from enum import Enum
import json
import os
from typing_extensions import Self

import numpy as np

from keras.src.utils import pad_sequences

from util.actlbl_image import ActlblImage

class ActlblActivity(Enum):
    Idle = 0
    Lying = 1
    Running = 2
    Sitting = 3
    StandingLying = 4
    StandingSitting = 5
    Walking = 6

    def from_str(name) -> Self:
        return list(filter(lambda x: x.name.lower() == name.lower(), ActlblActivity))[0]

    def load_history_group(self, root_directory):
        history_group = []
        group_directory = os.path.join(root_directory, self.name.lower())

        for history_filename in os.listdir(group_directory):
            history_filepath = os.path.join(group_directory, history_filename)
            
            with open(history_filepath, "r") as json_file:
                json_data = json.load(json_file)
                history = json_data["history"]
                history_group.append(np.array(history))
        
        n_average_frame = sum([len(i) for i in history_group]) / len(history)

        print(f"{group_directory}: {n_average_frame}")

        return history_group

def load_dataset(dataset_path: str):
    """
    Loads the given dataset directory. There should be a subdirectory for each action class inside the dataset folder.
    """

    train_data = []
    labels = []

    for (class_index, pathname) in enumerate(os.listdir(dataset_path)):
        canonical_path = os.path.join(dataset_path, pathname)

        if not os.path.isdir(canonical_path):
            continue

        activity = ActlblActivity.from_str(pathname)
        
        history_group = activity.load_history_group(dataset_path)

        for history in history_group:
            train_data.append(history)
            labels.append(class_index)

    return train_data, labels

def augment_dataset(history_list, labels, N=30, max_shift=0.01):
    new_history_list = []
    new_labels = []
    for history_index, history in enumerate(history_list):
        for i in range(N):
            new_history = []
            for frame in history:
                new_frame = []
                for keypoint in frame:
                    x, y, confidence = keypoint
                    new_keypoint = np.array([x, y, confidence])
                    new_keypoint += np.random.uniform(0, max_shift, 3)
                    new_frame.append(new_keypoint)
                new_history.append(new_frame)
            new_history_list.append(new_history)
            new_labels.append(labels[history_index])
    return new_history_list, new_labels

def history_add_padding(history_list):
    padded_inputs = pad_sequences(history_list, padding="post")

    return padded_inputs

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

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, Embedding
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MultiLabelBinarizer

import settings
from preprocessing.manager import history_add_padding, Manager
from settings import is_debug

# from movenet import filter_not

FRAME_PER_DATA = settings.get_frames_per_sample()  # Frame refers to a single movenet keypoints list
TIME_STEPS = 51 * FRAME_PER_DATA  # There are x, y coordinates for each 11 keypoints
N_FEATURES = 1


# TODO: 머신러닝 이용해서 외삽 진행합시다.
def preprocess_data(history: np.ndarray):
    """Preprocess history data before passing it to the model
        Args:
            history: History data to pass

        Returns:
            Processed history data
    """

    # filtered_indices = filter_not(['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'])
    # history = np.delete(history, filtered_indices, axis=1)
    # history = np.delete(history, [2], axis=2)  # TODO: remove confidence?

    return history


def load_dataset(directory: str):
    """Loads JSON dataset from given directory

    Converts JSON file into python dictionary.

    Args:
        directory: Name of the directory containing json files with history data
    Returns:
        Dictionary with history data
    """

    list_dataset = []

    files = os.listdir(directory)
    for json_dataset_path in files:
        json_full_path = os.path.join(directory, json_dataset_path)
        with open(json_full_path, "r") as json_file:
            json_data = json.load(json_file)
            history = json_data["history"]
            history = np.array(history)

            history = preprocess_data(history)
            print(f"{json_full_path} : {len(history)}")

            list_dataset.append(history)
    return list_dataset


def train_labeler(classified_history):
    train_data = [history for history_group in classified_history for history in history_group]
    labels = [group_id for (group_id, history_group) in enumerate(classified_history) for history in history_group]
    print(labels)
    manager = Manager(np.array(history_add_padding(train_data)), np.array(labels))
    accuracy = manager.predict()

    print(f"Accuracy: {accuracy}")

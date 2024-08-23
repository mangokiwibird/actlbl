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
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MultiLabelBinarizer

import settings
from settings import is_debug
# from movenet import filter_not

FRAME_PER_DATA = 25  # Frame refers to a single movenet keypoints list
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

            list_dataset.append(history)
    
    return list_dataset


def train_labeler(classified_history):
    labels = [group_id for (group_id, history_group) in enumerate(classified_history) for history in history_group]
    mlb = MultiLabelBinarizer()
    labels_for_training = mlb.fit_transform(np.array(labels).reshape(-1, 1))

    # Limits history size to frame_per_data for the consistency of data size
    raw_train_data = \
        np.array(
            [np.array(history[-FRAME_PER_DATA:]) for history_group in classified_history for history in history_group])

    n_samples = len(raw_train_data)

    # Conform to Conv1D input format
    flattened_train_data = np.zeros((n_samples, TIME_STEPS, N_FEATURES))
    for i, d in enumerate(raw_train_data):
        # TODO: currently flattened all x, y coordinates for Conv1D -> Use Conv2D?
        flattened_train_data[i, :, 0] = d.flatten()

    x_train, x_test, y_train, y_test = train_test_split(
        flattened_train_data,
        labels_for_training,
        test_size=0.3,
        random_state=42)

    model = Sequential([
        Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(TIME_STEPS, N_FEATURES)),
        MaxPooling1D(pool_size=2),
        Dropout(0.5),
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dropout(0.5),
        Dense(5, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=500, batch_size=64, validation_data=(x_test, y_test))

    if is_debug():
        print(history.history["accuracy"])
        print(history.history["val_accuracy"])
        print(history.history["loss"])
        print(history.history["val_loss"])
        # plt.figure(figsize=(12, 4))
        # plt.subplot(1, 2, 1)
        # plt.plot(history.history['accuracy'])
        # plt.plot(history.history['val_accuracy'])
        # plt.title('Model accuracy')
        # plt.ylabel('Accuracy')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Test'], loc='upper left')
        #
        # plt.subplot(1, 2, 2)
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('Model loss')
        # plt.ylabel('Loss')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Test'], loc='upper left')
        # plt.tight_layout()
        # plt.show()

    model.save(settings.get_target_model_path())

    return model

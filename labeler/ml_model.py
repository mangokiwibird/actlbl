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

from dataset_manager import augment_dataset, load_history_dataset, history_add_padding
import numpy as np
from keras.src.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras import Sequential

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam

import settings

def train_model(history_data, labels):
    data, labels = np.array(history_data), np.array(labels)

    one_hot_labels = to_categorical(labels, num_classes=len(set(labels)))

    train_x, val_x, train_y, val_y = train_test_split(data, np.array(one_hot_labels), test_size=0.2, random_state=31)

    model = Sequential([
        Conv2D(16, 3, padding="same", activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, padding="same", activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding="same", activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(set(labels)), activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_x, train_y, epochs=20, batch_size=64, validation_data=(val_x, val_y))
    model.summary()

    if settings.is_debug():
        plt.figure(figsize=(12, 4))
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Accuracy Function')
        plt.ylabel('Value')
        plt.xlabel('Epoch')
        plt.legend(['TrainA', 'TestA'], loc='upper left')
        plt.tight_layout()
        plt.show()

        tf.keras.utils.plot_model(model, show_shapes=True, to_file='pre_model.png')

    return model

def generate_model(dataset_directory):
    history_list, labels = load_history_dataset(dataset_directory)
    history_list, labels = augment_dataset(history_list, labels)
    history_list = history_add_padding(history_list)
    model = train_model(np.array(history_list), np.array(labels))
    return model

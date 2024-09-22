import random

import numpy as np
from keras.src.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv3D, Dense, Dropout
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Masking

from preprocessing import masking


class Manager:
    def __init__(self, history_data, answers):
        self.history_data = history_data  # Note: RAW history data must be given
        self.answers = answers

    def calculate_size(self):
        padded_data = masking.history_add_padding(self.history_data)
        masking_layer = masking.create_mask_layer(padded_data)

        feature_size = masking_layer.shape[2] * (masking_layer.shape[3] - 1)  # this case 51 - x, y, confidence for 17 parts
        time_steps = masking_layer.shape[1]  # data per sequence
        batch_size = masking_layer.shape[0]
        print(masking_layer.shape)

        return batch_size, feature_size, time_steps

    def process_data(self):
        list_history = self.history_data

        data_to_feed = []
        for history in list_history:
            sequence_data = []  # This sequence data will be the timestep
            for time_step in history:
                feature_list = []
                std_first = -1
                std_last = -1
                for movenet_unit in time_step:
                    time_step = []
                    if std_first == -1:
                        std_first = movenet_unit[0]
                    if std_last == -1:
                        std_last = movenet_unit[1]
                    time_step.append(movenet_unit[0])  # Append x
                    time_step.append(movenet_unit[1])  # Append y
                    time_step.append(movenet_unit[2])  # Append z
                    feature_list.append(time_step)
                sequence_data.append(feature_list)
            data_to_feed.append(sequence_data)
        return data_to_feed, self.answers

    def generate_model(self):
        batch_size, feature_size, time_steps = self.calculate_size()
        data, answers = self.process_data()

        # Convert python list to numpy
        data = np.array(data)
        answers = np.array(answers)

        one_hot_labels = to_categorical(answers, num_classes=4)

        num_sequences, num_frames, num_keypoints, features = data.shape
        data = data.reshape((num_sequences, num_frames, num_keypoints, 1, features))

        # import sys
        # import numpy
        # numpy.set_printoptions(threshold=sys.maxsize)
        #
        # print(data)

        # Scale data - Flatten and reshape due to dimension limit of StandardScaler
        # data_reshaped = data.reshape(-1, data.shape[2])
        # scaler = StandardScaler()
        # scaled_data_reshaped = scaler.fit_transform(data_reshaped)
        # scaled_data = scaled_data_reshaped.reshape(data.shape)

        # Train Test Split
        train_x, val_x, train_y, val_y = train_test_split(np.array(data), np.array(one_hot_labels), test_size=0.2, random_state=34)

        model = Sequential()

        # Masking
        # model.add(Masking(0.0, input_shape=(num_frames, num_keypoints, 1, features)))

        # Convolution Layers
        model.add(Conv3D(32, kernel_size=(3, 3, 1), activation='relu', padding='same', input_shape=(num_frames, num_keypoints, 1, features)))
        model.add(MaxPooling3D(pool_size=(2, 2, 1)))
        model.add(Dropout(0.4))

        # Flatten for Fully Connected Layers
        model.add(Flatten())

        # Dense Layers
        model.add(Dense(32, activation='sigmoid'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='sigmoid'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='sigmoid'))
        model.add(Dense(4, activation='softmax'))

        model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        history = model.fit(train_x, train_y, epochs=200, batch_size=10, validation_data=(val_x, val_y))

        # Visualize actual results?
        res = [model.predict(np.array([data[i]])) for i in range(len(answers))]
        print(res)
        for i, x in enumerate(res):
            x = x[0]
            y = max(x[0], x[1], x[2], x[3])
            if y == x[0]:
                print(0 == answers[i])
            elif y == x[1]:
                print(1 == answers[i])
            elif y == x[2]:
                print(2 == answers[i])
            else:
                print(3 == answers[i])


        plt.figure(figsize=(12, 4))

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Graph of Everything')
        plt.ylabel('Value')
        plt.xlabel('Epoch')
        plt.legend(['TrainA', 'TestA'], loc='upper left')
        plt.tight_layout()
        plt.show()

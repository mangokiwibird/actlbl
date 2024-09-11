import random

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Masking
from tensorflow.keras.optimizers import SGD

from preprocessing import masking


class Manager:
    def __init__(self, history_data, answers):
        self.history_data = history_data  # Note: RAW history data must be given
        self.answers = answers

    def calculate_size(self):
        padded_data = masking.history_add_padding(self.history_data)
        masking_layer = masking.create_mask_layer(padded_data)

        feature_size = masking_layer.shape[2] * masking_layer.shape[3]  # this case 51 - x, y, confidence for 17 parts
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
                for movenet_unit in time_step:
                    feature_list.append(movenet_unit[0] + random.uniform(0.0, 3.0) - 1.5)  # Append x
                    feature_list.append(movenet_unit[1] + random.uniform(0.0, 3.0) - 1.5)  # Append y
                    feature_list.append(movenet_unit[2] + random.uniform(0.0, 3.0) - 1.5)  # Append z
                sequence_data.append(feature_list)
            data_to_feed.append(sequence_data)
        return data_to_feed, self.answers

    def generate_model(self):
        batch_size, feature_size, time_steps = self.calculate_size()
        data, answers = self.process_data()

        # Convert python list to numpy
        data = np.array(data)
        answers = np.array(answers)

        # Scale data - Flatten and reshape due to dimension limit of StandardScaler
        data_reshaped = data.reshape(-1, data.shape[2])
        scaler = StandardScaler()
        scaled_data_reshaped = scaler.fit_transform(data_reshaped)
        scaled_data = scaled_data_reshaped.reshape(data.shape)

        # Train Test Split
        train_x, val_x, train_y, val_y = train_test_split(np.array(scaled_data), np.array(answers), test_size=0.4, random_state=34)

        model = Sequential()

        # Masking
        model.add(Masking(0.0, input_shape=(time_steps, feature_size)))

        # Convolution Layers
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(time_steps, feature_size)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.3))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2, padding="same"))
        model.add(Dropout(0.2))

        # Flatten for Fully Connected Layers
        model.add(Flatten())

        # Dense Layers
        model.add(Dense(512, activation='sigmoid'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='sigmoid'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='sigmoid'))
        model.add(Dense(1, activation='relu'))

        model.compile(optimizer="adam", loss='mean_squared_error', metrics=['mae'])
        model.summary()

        history = model.fit(train_x, train_y, epochs=3000, batch_size=10, validation_data=(val_x, val_y))

        # Visualize actual results?
        res = [round(float(model.predict(np.array([scaled_data[i]]))[0])) for i in range(len(answers))]
        print(res)

        plt.figure(figsize=(12, 4))

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title('Graph of Everything')
        plt.ylabel('Value')
        plt.xlabel('Epoch')
        plt.legend(['TrainL', 'TestL', 'TrainA', 'TestA'], loc='upper left')
        plt.tight_layout()
        plt.show()

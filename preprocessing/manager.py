import numpy as np
from keras.src.utils import to_categorical, pad_sequences
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv3D, Dense, Dropout
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import Flatten


def history_add_padding(list_history):
    padded_inputs = pad_sequences(list_history, padding="post")
    return padded_inputs


class Manager:
    def __init__(self, history_data, answers):
        self.history_data = history_data  # Note: RAW history data must be given
        self.answers = answers

    def generate_model(self):
        data, answers = self.history_data, self.answers

        # Convert python list to numpy
        data = np.array(data)
        answers = np.array(answers)

        one_hot_labels = to_categorical(answers, num_classes=len(set(self.answers)))

        num_sequences, num_frames, num_keypoints, features = data.shape
        data = data.reshape((num_sequences, num_frames, num_keypoints, 1, features))

        # Train Test Split
        train_x, val_x, train_y, val_y = train_test_split(np.array(data), np.array(one_hot_labels), test_size=0.2, random_state=34)

        model = Sequential()

        # Convolution Layers
        model.add(Conv3D(32, kernel_size=(3, 3, 1), activation='relu', padding='same', input_shape=(num_frames, num_keypoints, 1, features)))
        model.add(MaxPooling3D(pool_size=(2, 2, 1)))
        model.add(Dropout(0.4))

        # Flatten for Fully Connected Layers
        model.add(Flatten())

        # Dense Layers
        model.add(Dense(128, activation='sigmoid'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='sigmoid'))
        model.add(Dense(4, activation='softmax'))

        model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        history = model.fit(train_x, train_y, epochs=200, batch_size=10, validation_data=(val_x, val_y))

        plt.figure(figsize=(12, 4))

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Graph of Everything')
        plt.ylabel('Value')
        plt.xlabel('Epoch')
        plt.legend(['TrainA', 'TestA'], loc='upper left')
        plt.tight_layout()
        plt.show()

        res = 0

        for i in range(len(answers)):
            predict_result = model.predict(np.array([data[i]]), verbose=False)[0].tolist()
            res += int(predict_result.index(max(predict_result)) == answers[i])

        return res / len(answers)

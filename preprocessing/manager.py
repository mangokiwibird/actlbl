from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout
from tensorflow.keras.layers import MaxPooling1D
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

        feature_size = masking_layer.shape[2] * masking_layer.shape[3]  # this case 51 - x, y, confidence for 17 parts
        time_steps = masking_layer.shape[1]  # data per sequence
        print(masking_layer.shape)

        return feature_size, time_steps

    def process_data(self):
        list_history = self.history_data

        data_to_feed = []
        for history in list_history:
            sequence_data = []  # This sequence data will be the timestep
            for time_step in history:
                feature_list = []
                for movenet_unit in time_step:
                    feature_list.append(movenet_unit[0])  # Append x
                    feature_list.append(movenet_unit[1])  # Append y
                    feature_list.append(movenet_unit[2])  # Append z
                sequence_data.append(feature_list)
            data_to_feed.append(sequence_data)
        return data_to_feed, self.answers

    def generate_model(self):
        feature_size, time_steps = self.calculate_size()
        print(feature_size, time_steps)
        data, answers = self.process_data()

        X_train, X_val, y_train, y_val = train_test_split(data, answers, test_size=0.2, random_state=42)

        model = Sequential()

        model.add(Masking(input_shape=(time_steps, feature_size)))

        # First 1D Convolutional layer
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(time_steps, feature_size)))

        # Max pooling layer to reduce dimensionality
        model.add(MaxPooling1D(pool_size=2))

        # Additional Convolutional layers (optional)
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))

        model.add(Flatten())

        # Fully connected Dense layers
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Summary of the model
        model.summary()

        # Training the model (assuming X_train and y_train are prepared)
        # X_train should have shape (samples, timesteps, features), and y_train is one-hot encoded
        history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Function Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

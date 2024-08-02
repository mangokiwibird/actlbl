import json
import numpy as np

from keras._tf_keras.keras.models import Sequential # todo is this right?
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout

class MLBasedLabeler:
    """
    Lables action using keypoints parsed by movenet.
    """

    history = []

    def append_keypoints_to_history(self, keypoints):
        """
        save frame to history
        """
        if len(self.history) == 100:
            self.history.pop(0)
        self.history.append(np.array(keypoints).tolist()[0][0])

    # TODO: implement
    def get_score(self, keypoints):
        """
        returns dictionary of [action - score]
        """

        self.append_keypoints_to_history(keypoints)

        # model = Sequential([
        #     LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
        #     Dropout(0.2),
        #     Dense(1, activation='sigmoid')
        # ])

        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # model.summary()

        return { "walking" : 1.0 }
    
    def save_data(self):
        """Saves history data to json file"""

        with open("./data.json", "w") as outfile:
            json.dump({ "history" : self.history }, outfile)
            print("successfully saved data to file")
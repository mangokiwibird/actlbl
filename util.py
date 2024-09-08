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

from matplotlib import pyplot as plt
import numpy as np


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

from settings import is_debug


class Extrapolation:
    """Class for extrapolation utilities

    Attributes:
        fed_data: data before extrapolation
        model: Trained model after train_model() call
    """

    def __init__(self, fed_data: dict[str, list]):
        """Extrapolates x, y coordinates(output) for a given keyframe index(input)

        Args:
            fed_data: data before extrapolation
        """

        self.model = None
        self.fed_data = fed_data

    def train_model(self):
        x_train, x_test, y_train, y_test = train_test_split(np.array(list(map(int, self.fed_data.keys()))), np.array(list(self.fed_data.values())) , test_size=0.3, random_state=42)
        
        model = Sequential([
            Dense(64, activation="relu"),
            Dense(128, activation="sigmoid"),
            Dense(32, activation="relu"),
            Dense(2, activation="linear")
        ])

        model.compile(optimizer="adam", loss='mean_squared_error', metrics=['mae'])
        model.summary()

        history = model.fit(x_train, y_train, epochs=500, batch_size=64, validation_data=(x_test, y_test))

        if is_debug():
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')

            plt.show()

        self.model = model


extrapolation = Extrapolation({
    "1": [1, 2],
    "100": [100, 200],
    "2": [2, 4],
    "4": [4, 8],
    "6": [6, 12],
    "10": [10, 20],
    "80": [80, 160],
    "18": [18, 36]
})

extrapolation.train_model()
print(extrapolation.model.predict(np.array([11])))

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from keras._tf_keras.keras.optimizers import Adam
from sklearn.preprocessing import MultiLabelBinarizer

from model import filter_not

# TODO: 머신러닝 이용해서 외삽 진행합시다.
def preprocess_data(history):
    filtered_indices = filter_not(['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'])
    history = np.delete(history, filtered_indices, axis=1)

    # preprocessed_data = [[ [[] for k in range(17-len(filtered_indices))] ] for h in range(len(history))]

    # for i in range(17 - len(filtered_indices)):
    #     last_idx = -1
    #     print(f"\n\n\n\n{i}")
    #     data = [(idx, movenet_frame[i]) for (idx, movenet_frame) in enumerate(history) if movenet_frame[i][2] > 0.1]
    #     # print(data)
    #     print(data)
    #     for idx, frame in data:
    #         print(f"->{last_idx + 1} ~ {idx + 1}")
    #         for j in range(last_idx + 1, idx + 1):
    #             print(f"{j}, {i}: {frame}")
    #             preprocessed_data[j][i] = frame
    #         last_idx = idx
            
    history = np.delete(history, [2], axis=2)

    return history

def load_dataset(directory: str):
    """
    Loads JSON dataset from given directory
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

dataset = [load_dataset("./model/꽁꽁"), load_dataset("./model/얼어붙은"), load_dataset("./model/한강위로"), load_dataset("./model/고양이가"), load_dataset("./model/걸어다닙니다")]

frame_per_data = 25
time_steps = 22 * frame_per_data
n_features = 1

labels = [i for (i, data_group) in enumerate(dataset) for history in data_group]

mlb = MultiLabelBinarizer()
labels_for_training = mlb.fit_transform(np.array(labels).reshape(-1, 1))

dataset_for_training = np.array([np.array(x[:frame_per_data]) for data in dataset for x in data])
n_samples = len(dataset_for_training)

data = np.random.randn(n_samples, time_steps, n_features)
for i, d in enumerate(dataset_for_training):
    data[i, :, 0] = d.flatten()

X_train, X_test, y_train, y_test = train_test_split(data, labels_for_training, test_size=0.3, random_state=42)

model = Sequential([
    Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(time_steps, n_features)),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),
    Conv1D(filters=64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(50, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=500, batch_size=64, validation_data=(X_test, y_test))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.tight_layout()
plt.show()

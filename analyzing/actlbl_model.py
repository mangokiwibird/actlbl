import numpy as np

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf

# TODO: Replace keras._tf_keras.keras to tf.keras
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras import Model
from keras._tf_keras.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Input, Concatenate, Layer
from keras._tf_keras.keras.optimizers import Adam

def build_model(num_class = 7):
    conv_layer1 = Conv2D(16, 3, padding="same", activation='relu')
    max_pooling_layer1 = MaxPooling2D()
    conv_layer2 = Conv2D(32, 3, padding="same", activation='relu')
    max_pooling_layer2 = MaxPooling2D()
    conv_layer3 = Conv2D(64, 3, padding="same", activation='relu')
    max_pooling_layer3 = MaxPooling2D()
    flatten_layer = Flatten()
    dense_layer1 = Dense(128, activation='relu')

    dense_layer2 = Dense(128, activation='relu')
    dense_layer3 = Dense(64, activation='relu')

    concat_layer1 = Concatenate()

    output_layer1 = Dense(num_class, activation='sigmoid')

    dense_layer4 = Dense(num_class, activation='sigmoid')
    
    concat_layer2 = Concatenate()
    dense_layer5 = Dense(128, activation='relu')
    output_layer2 = Dense(num_class, activation='sigmoid')

    # ---

    movenet_input = Input() # todo define size
    features_input = Input() # todo define size
    history_input = Input() # todo define size

    conv1 = conv_layer1(movenet_input)
    max_pooling1 = max_pooling_layer1(conv1)
    conv2 = conv_layer2(max_pooling1)
    max_pooling2 = max_pooling_layer2(conv2)
    conv3 = conv_layer3(max_pooling2)
    max_pooling3 = max_pooling_layer3(conv3)
    flatten = flatten_layer(max_pooling3)
    dense1 = dense_layer1(flatten)

    dense2 = dense_layer2(features_input)
    dense3 = dense_layer3(dense2)

    concat1 = concat_layer1([dense1, dense3])

    output1 = output_layer1(concat1)

    dense4 = dense_layer4(concat1)

    concat2 = concat_layer2([dense4, history_input])
    dense5 = dense_layer5(concat2)
    output2 = output_layer2(dense5)

    return movenet_input, features_input, history_input, output1, output2
    
    
def train_model(history_data, labels):
    movenet_input, features_input, history_input, output1, output2 = build_model()
    model = Model(inputs = [movenet_input, features_input, history_input], outputs=[output1, output2])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    data, labels = np.array(history_data), np.array(labels)
    one_hot_labels = to_categorical(labels, num_classes=len(set(labels)))
    train_x, val_x, train_y, val_y = train_test_split(data, np.array(one_hot_labels), test_size=0.2, random_state=31)

    fake_features_x = np.copy(train_x)
    fake_history_x = np.copy(train_x)
    for i in range(len(fake_features_x)):
        fake_features_x[i] = np.array([10.0, 10.0, 10.0])
        fake_history_x[i] = train_y[i]

    val_features_x = np.copy(train_x)
    val_history_x = np.copy(train_x)
    for i in range(len(fake_features_x)):
        val_features_x[i] = np.array([10.0, 10.0, 10.0])
        val_history_x[i] = val_y[i]

    history = model.fit(
        (train_x, fake_features_x, fake_history_x), 
        (train_y, train_y),
        epochs=200, 
        batch_size=64, 
        validation_data=(
            (val_x, val_features_x, val_history_x), 
            (val_y, val_y)))
    
    model.summary()
    
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    return model
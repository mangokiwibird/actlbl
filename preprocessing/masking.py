import tensorflow as tf

from tensorflow.keras.layers import Masking


class PaddedData:
    def __init__(self, masked_data):
        self.data = masked_data

    def get_data_shape(self):
        return self.data.shape


def history_add_padding(list_history):
    padded_inputs = tf.keras.utils.pad_sequences(list_history, padding="post")
    return PaddedData(padded_inputs)


def create_mask_layer(padded_history: PaddedData): # TODO don't know what this does
    masking_layer = Masking(0.0)
    return masking_layer(padded_history.data)

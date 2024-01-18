import os
import tensorflow as tf
from cryptography.fernet import Fernet

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


class Conv2Plus1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding, data_format):
        """
          A sequence of convolutional layers that first apply the convolution operation over the
          spatial dimensions, and then the temporal dimension.
        """
        super().__init__()
        self.seq = tf.keras.models.Sequential([
            # Spatial decomposition
            tf.keras.layers.Conv3D(filters=filters,
                                   kernel_size=(1, kernel_size[1], kernel_size[2]),
                                   padding=padding, data_format=data_format),
            # Temporal decomposition
            tf.keras.layers.Conv3D(filters=filters,
                                   kernel_size=(kernel_size[0], 1, 1),
                                   padding=padding, data_format=data_format)
            ])

    def call(self, x):
        return self.seq(x)


def build_rnn_model():
    fast_image_input = tf.keras.layers.Input(shape=(None, 64, 64, 1), name='fast_input')
    fast_timestamp_input = tf.keras.layers.Input(shape=(None, 2))
    fast_rnn = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(fast_image_input)
    fast_rnn = tf.keras.layers.MaxPooling3D((1, 2, 2))(fast_rnn)
    fast_rnn = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(fast_rnn)
    fast_rnn = tf.keras.layers.MaxPooling3D((1, 2, 2))(fast_rnn)
    fast_rnn = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(fast_rnn)
    fast_rnn = tf.keras.layers.MaxPooling3D((1, 2, 2))(fast_rnn)
    fast_rnn = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(fast_rnn)
    fast_rnn = tf.keras.layers.LSTM(128, return_sequences=True)(fast_rnn)

    fast_timestamp_rnn = tf.keras.layers.LSTM(64, return_sequences=True)(fast_timestamp_input)

    fast_features = tf.keras.layers.Concatenate()([fast_rnn, fast_timestamp_rnn])
    fast_lstm = tf.keras.layers.LSTM(128, return_sequences=False)(fast_features)

    #####################################################################################

    slow_rnn = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(fast_image_input)
    slow_rnn = tf.keras.layers.MaxPooling3D((2, 2, 2))(slow_rnn)
    slow_rnn = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(slow_rnn)
    slow_rnn = tf.keras.layers.MaxPooling3D((2, 2, 2))(slow_rnn)
    slow_rnn = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(slow_rnn)
    slow_rnn = tf.keras.layers.MaxPooling3D((2, 2, 2))(slow_rnn)
    slow_rnn = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(slow_rnn)
    slow_rnn = tf.keras.layers.LSTM(128, return_sequences=True)(slow_rnn)

    # slow_timestamp_rnn = tf.keras.layers.LSTM(128, return_sequences=True)(slow_timestamp_input)

    # slow_features = tf.keras.layers.Concatenate()([slow_rnn, slow_timestamp_rnn])
    slow_lstm = tf.keras.layers.LSTM(64, return_sequences=False)(slow_rnn)

    fused_features = tf.keras.layers.Concatenate()([fast_lstm, slow_lstm])

    # Fully connected layers for classification
    fc1 = tf.keras.layers.Dense(64, activation='relu')(fused_features)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(fc1)

    # Create the model
    model = tf.keras.models.Model(inputs=[fast_image_input, fast_timestamp_input], outputs=output)

    return model


# Encrypt the model weights
def encrypt_model(model_name, key=b'J17tdv3zz2nemLNwd17DV33-sQbo52vFzl2EOYgtScw='):
    # Read the entire model file
    with open(f"{model_name}.h5", "rb") as file:
        model_bytes = file.read()

    # Use the provided key to create a cipher
    cipher = Fernet(key)

    # Encrypt the entire model
    encrypted_model = cipher.encrypt(model_bytes)

    # Save the encrypted weights to a file
    with open(f"{model_name}.bin", "wb") as file:
        file.write(encrypted_model)


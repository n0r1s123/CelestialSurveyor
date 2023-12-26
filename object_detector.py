import os
import tensorflow as tf
from dataset_creator.dataset import SourceData
from dataset_creator.training_dataset import TrainingDataset
from cryptography.fernet import Fernet

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
#
#
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



# Define your RNN model
def build_rnn_model(input_shape):
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



    # fast_image_input = tf.keras.layers.Input(shape=(None, 64, 64, 1), name='fast_input')
    # slow_image_input = tf.keras.layers.Input(shape=(None, 64, 64, 1), name='slow_input')
    # timestamp_input = tf.keras.layers.Input(shape=(None, 2))
    #
    # fast_rnn = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'))(fast_image_input)
    # fast_rnn = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2)))(fast_rnn)
    # fast_rnn = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(23, (3, 3), activation='relu', padding='same'))(fast_rnn)
    # fast_rnn = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2)))(fast_rnn)
    # fast_rnn = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))(fast_rnn)
    # fast_rnn = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2)))(fast_rnn)
    # fast_rnn = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(fast_rnn)
    # image_lstm = tf.keras.layers.LSTM(128, return_sequences=False)(fast_rnn)
    #
    # # Slow modality branch
    # slow_rnn = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'))(slow_image_input)
    # slow_rnn = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2)))(slow_rnn)
    # slow_rnn = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))(slow_rnn)
    # slow_rnn = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2)))(slow_rnn)
    # slow_rnn = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))(slow_rnn)
    # slow_rnn = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2)))(slow_rnn)
    # slow_rnn = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(slow_rnn)
    # slow_lstm = tf.keras.layers.LSTM(128, return_sequences=False)(slow_rnn)
    #
    # timestamp_rnn = tf.keras.layers.LSTM(32, return_sequences=False)(timestamp_input)
    # fused_features = tf.keras.layers.Concatenate()([image_lstm, slow_lstm, timestamp_rnn])
    #
    # # Fully connected layers for classification
    # fc1 = tf.keras.layers.Dense(64, activation='relu')(fused_features)
    # output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(fc1)
    #
    # # Create the model
    # model = tf.keras.models.Model(inputs=[fast_image_input, slow_image_input, timestamp_input], outputs=output)


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


def main():
    print(tf.__version__)
    input_shape = (None, 64, 64, 1)
    load_model_name = "model42"
    save_model_name = "model43"


    # Build the model

    # Compile the model
    # model = build_rnn_model(input_shape)
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Load model
    model = tf.keras.models.load_model(
        f'{load_model_name}.h5'
    )

    print(model.summary())
    dataset = TrainingDataset([
            # SourceData('C:\\Users\\bsolomin\\Astro\\SeaHorse\\cropped\\', non_linear=True, to_align=False),
            # SourceData('C:\\Users\\bsolomin\\Astro\\SeaHorse\\cropped\\', non_linear=True, num_from_session=5),
            # SourceData('C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\cropped', non_linear=True, to_align=False),
            SourceData('C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\cropped', non_linear=True, num_from_session=10, to_align=False),
            # SourceData('C:\\Users\\bsolomin\\Astro\\Andromeda\\Pix_600\\cropped\\', non_linear=True, to_align=False),
            # SourceData('C:\\Users\\bsolomin\\Astro\\Andromeda\\Pix_600\\cropped\\', non_linear=True, num_from_session=10, to_align=False),
            SourceData('C:\\Users\\bsolomin\\Astro\\Orion\\Part_four\\cropped1\\', non_linear=True, to_align=False),
            SourceData('C:\\Users\\bsolomin\\Astro\\Orion\\Part_four\\cropped1\\', non_linear=True, num_from_session=5),
            SourceData('C:\\Users\\bsolomin\\Astro\\Orion\\Part_one\\cropped\\', non_linear=True, to_align=False),
            SourceData('C:\\Users\\bsolomin\\Astro\\Orion\\Part_three\\cropped\\', non_linear=True, to_align=False),
            SourceData('C:\\Users\\bsolomin\\Astro\\Orion\\Part_three\\cropped\\', non_linear=True, num_from_session=5),
            SourceData('C:\\Users\\bsolomin\\Astro\\Orion\\Part_two\\cropped\\', non_linear=True, to_align=False),
            # # SourceData('C:\\Users\\bsolomin\\Astro\\M81\\cropped\\', non_linear=True),
            # # SourceData('C:\\Users\\bsolomin\\Astro\\NGC_1333_RASA\\cropped', non_linear=True, num_from_session=5),
            SourceData('C:\\Users\\bsolomin\\Astro\\NGC_1333_RASA\\cropped', non_linear=True, to_align=False),
            SourceData('D:\\Boris\\astro\\Auriga\\Light', non_linear=False, to_align=True, to_skip_bad=False, num_from_session=20),

        ],
        samples_folder='C:\\git\\object_recognition\\star_samples',
    )

    training_generator = dataset.batch_generator(batch_size=10)
    val_generator = dataset.batch_generator(batch_size=10)

    early_stopping_monitor = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=15,
        verbose=1,
        mode='min',
        baseline=None,
        restore_best_weights=True
    )

    try:
        model.fit(
            training_generator,
            validation_data=val_generator,
            steps_per_epoch=5000,
            validation_steps=2000,
            epochs=40,
            callbacks=[early_stopping_monitor]
        )
    except KeyboardInterrupt:
        model.save(f"{save_model_name}.h5")
        encrypt_model(save_model_name)

    model.save(f"{save_model_name}.h5")
    encrypt_model(save_model_name)


if __name__ == '__main__':
    main()


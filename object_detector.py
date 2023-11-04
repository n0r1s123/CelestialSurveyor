import os
import tensorflow as tf
from dataset_creator.dataset_creator import SourceData, Dataset

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


# Define your RNN model
def build_rnn_model(input_shape):
    # Define your CNN and RNN models
    # cnn_model = tf.keras.models.Sequential([
    #     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(54, 54, 1)),
    #     tf.keras.layers.MaxPooling2D((2, 2)),
    #     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    #     tf.keras.layers.MaxPooling2D((2, 2)),
    #
    #     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    #     tf.keras.layers.MaxPooling2D((2, 2)),
    #
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.BatchNormalization(),
    # ])
    #
    # rnn_model = tf.keras.models.Sequential([
    #     # tf.keras.layers.LSTM(128, return_sequences=True),  # LSTM layer with 64 units
    #     tf.keras.layers.LSTM(256, return_sequences=False),  # LSTM layer with 32 units
    # ])
    #
    # # Define the input layers for both the images and timestamps
    # image_input = tf.keras.layers.Input(shape=(None, 54, 54, 1))
    # timestamp_input = tf.keras.layers.Input(shape=(None, 1))
    #
    # # Use the CNN model for processing images
    # image_features = tf.keras.layers.TimeDistributed(cnn_model)(image_input)
    #
    # # Concatenate the processed image features with the timestamps
    # combined_features = tf.keras.layers.Concatenate()([image_features, timestamp_input])
    #
    # # Use the RNN model for processing the combined features
    # rnn_output = rnn_model(combined_features)
    #
    # dense_output = tf.keras.layers.Dense(256, activation='relu')(rnn_output)
    #
    # # Add dense layers for classification
    # output = tf.keras.layers.Dense(1, activation='sigmoid')(dense_output)
    #
    # # Create the model
    # model = tf.keras.models.Model(inputs=[image_input, timestamp_input], outputs=output)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), input_shape=(None, 54, 54, 1), padding='same',
               activation='relu'),
        tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same',
               activation='relu'),
        tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),
        tf.keras.layers.Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same',
               activation='relu'),
        tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ConvLSTM2D(filters=256, kernel_size=(2, 2), padding='same',
                   return_sequences=False),
        tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=False),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),  # Flatten the output before passing it through Dense layers
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output a single node with sigmoid activation for binary classification
    ])

    # model = tf.keras.models.Sequential()
    #
    # # Add a 3D convolutional layer
    # model.add(tf.keras.layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu',
    #                  input_shape=(None, 54, 54, 1)))
    #
    # # Add a 3D max pooling layer
    # model.add(tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)))
    #
    # # Flatten the output for dense layers
    # model.add(tf.keras.layers.Flatten())
    #
    # # Add one or more dense layers for classification
    # model.add(tf.keras.layers.Dense(64, activation='relu'))
    # model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Output layer (binary classification)

    return model


if __name__ == '__main__':
    print(tf.__version__)
    input_shape = (None, 54, 54, 1)
    # Build the model

    # Compile the model
    # model = build_rnn_model(input_shape)
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Load model
    model = tf.keras.models.load_model(
        'model14.h5'
    )

    print(model.summary())
    source_data = SourceData(
        [
            # 'C:\\Users\\bsolomin\\Astro\\Andromeda\\Pix_600\\cropped',
            # 'C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\cropped',
            # 'C:\\Users\\bsolomin\\Astro\\SeaHorse\\cropped',
            'C:\\Users\\bsolomin\\Astro\\NGC_1333_RASA\\cropped\\',
         ],
        samples_folder='C:\\git\\object_recognition\\star_samples')

    dataset = Dataset(source_data)

    # training_generator = DataGenerator(dataset, range(80000), range(80000), batch_size=10)
    # val_generator = DataGenerator(dataset, range(5000), range(5000), batch_size=10)
    training_generator = dataset.batch_generator(batch_size=10)
    val_generator = dataset.batch_generator(batch_size=10)
    try:
        model.fit(
            training_generator,
            validation_data=val_generator,
            steps_per_epoch=1000,
            validation_steps=100,
            epochs=15,
        )
    except KeyboardInterrupt:
        model.save("model15.h5")
    model.save("model15.h5")
    # finally:
    #     pass


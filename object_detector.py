import os
import tensorflow as tf
from dataset_creator.dataset_creator import SourceData, Dataset, DataGenerator


os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


# Define your RNN model
def build_rnn_model(input_shape):
    # Define your CNN and RNN models
    cnn_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(54, 54, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
    ])

    rnn_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(512, return_sequences=True),  # LSTM layer with 64 units
        tf.keras.layers.LSTM(128, return_sequences=False),  # LSTM layer with 32 units
    ])

    # Define the input layers for both the images and timestamps
    image_input = tf.keras.layers.Input(shape=(None, 54, 54, 1))
    timestamp_input = tf.keras.layers.Input(shape=(None, 1))

    # Use the CNN model for processing images
    image_features = tf.keras.layers.TimeDistributed(cnn_model)(image_input)

    # Concatenate the processed image features with the timestamps
    combined_features = tf.keras.layers.Concatenate()([image_features, timestamp_input])

    # Use the RNN model for processing the combined features
    rnn_output = rnn_model(combined_features)

    # Add dense layers for classification
    output = tf.keras.layers.Dense(1, activation='sigmoid')(rnn_output)

    # Create the model
    model = tf.keras.models.Model(inputs=[image_input, timestamp_input], outputs=output)

    return model


if __name__ == '__main__':
    print(tf.__version__)
    input_shape = (None, 54, 54, 1)
    # Build the model

    # Compile the model
    # model = build_rnn_model(input_shape)
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.build()


    # Load model
    model = tf.keras.models.load_model(
        'model11.h5'
    )

    print(model.summary())
    source_data = SourceData(
        'C:\\Users\\bsolomin\\Astro\\Andromeda\\Pix_600\\cropped',
        # 'C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\cropped',
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
            steps_per_epoch=10000,
            validation_steps=1000,
            max_queue_size=30,
            epochs=5,
        )
    except KeyboardInterrupt:
        model.save("model11.h5")
    model.save("model11.h5")


import os
import tensorflow as tf
from dataset_creator.dataset_creator import SourceData, Dataset, DataGenerator


os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


# Define your RNN model
def build_rnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
        #
        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu')),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),

        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, (3, 3), activation='relu')),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),

        #
        # tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128), input_shape=input_shape),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten(), input_shape=input_shape),

        # tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(128, return_sequences=False),

        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
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
        'model7.h5'
    )

    print(model.summary())
    source_data = SourceData(
        folder='C:\\Users\\bsolomin\\Astro\\Andromeda\\Pix_600\\cropped\\',
        samples_folder='C:\\git\\object_recognition\\star_samples')

    dataset = Dataset(source_data)

    training_generator = DataGenerator(dataset, range(80000), range(80000), batch_size=10)
    val_generator = DataGenerator(dataset, range(5000), range(5000), batch_size=10)
    try:
        model.fit(
            training_generator,
            validation_data=val_generator,
            epochs=5,
        )
    except KeyboardInterrupt:
        model.save("model7.h5")
    model.save("model7.h5")


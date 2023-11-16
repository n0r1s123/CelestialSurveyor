import os
import tensorflow as tf
from dataset_creator.dataset_creator import SourceData, Dataset

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


# Define your RNN model
def build_rnn_model(input_shape):
    image_input = tf.keras.layers.Input(shape=(None, 54, 54, 1))
    timestamp_input = tf.keras.layers.Input(shape=(None, 2))

    first_part = tf.keras.models.Sequential([
        tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), input_shape=(None, 54, 54, 1), padding='same',
               data_format="channels_last"),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same',
               data_format="channels_last"),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same',
               data_format="channels_last"),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
    ])

    second_part = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(256, return_sequences=False),
        tf.keras.layers.Dense(128),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dense(64),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    image_features = first_part(image_input)
    combined_features = tf.keras.layers.Concatenate()([image_features, timestamp_input])
    output = second_part(combined_features)
    model = tf.keras.models.Model(inputs=[image_input, timestamp_input], outputs=output)

    print(first_part.summary())
    print(second_part.summary())


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
        'model23.h5'
    )

    print(model.summary())
    source_data = SourceData(
        [
            # 'C:\\Users\\bsolomin\\Astro\\SeaHorse\\cropped\\',
            # 'C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\cropped',
            # 'C:\\Users\\bsolomin\\Astro\\Andromeda\\Pix_600\\cropped\\',
            # 'C:\\Users\\bsolomin\\Astro\\NGC_1333_RASA\\cropped\\',
            'C:\\Users\\bsolomin\\Astro\\Orion\\Part_four\\cropped\\',
            'C:\\Users\\bsolomin\\Astro\\Orion\\Part_one\\cropped\\',
            'C:\\Users\\bsolomin\\Astro\\Orion\\Part_two\\cropped\\',
            'C:\\Users\\bsolomin\\Astro\\Orion\\Part_three\\cropped\\',
            'C:\\Users\\bsolomin\\Astro\\M81\\cropped\\',
         ],
        samples_folder='C:\\git\\object_recognition\\star_samples')

    dataset = Dataset(source_data)

    training_generator = dataset.batch_generator(batch_size=10)
    val_generator = dataset.batch_generator(batch_size=10)
    try:
        model.fit(
            training_generator,
            validation_data=val_generator,
            steps_per_epoch=3000,
            validation_steps=500,
            epochs=3,
        )
    except KeyboardInterrupt:
        model.save("model24.h5")
    model.save("model24.h5")

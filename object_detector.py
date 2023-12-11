import os
import tensorflow as tf
from dataset_creator.dataset import SourceData
from dataset_creator.training_dataset import TrainingDataset
from cryptography.fernet import Fernet

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


# Define your RNN model
def build_rnn_model(input_shape):
    image_input = tf.keras.layers.Input(shape=(None, 64, 64, 1))
    timestamp_input = tf.keras.layers.Input(shape=(None, 2))

    first_part = tf.keras.models.Sequential([
        tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), input_shape=(None, 64, 64, 1), padding='same',
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
        tf.keras.layers.LSTM(256, return_sequences=True),
        tf.keras.layers.LSTM(128, return_sequences=False),
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
    load_model_name = "model31"
    save_model_name = "model32"


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
            # SourceData('C:\\Users\\bsolomin\\Astro\\SeaHorse\\cropped\\', non_linear=True),
            # SourceData('C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\cropped', non_linear=True),
            # SourceData('C:\\Users\\bsolomin\\Astro\\Andromeda\\Pix_600\\cropped\\', non_linear=True),
            # SourceData('C:\\Users\\bsolomin\\Astro\\Orion\\Part_four\\cropped1\\', non_linear=True),
            # SourceData('C:\\Users\\bsolomin\\Astro\\Orion\\Part_one\\cropped\\', non_linear=True),
            # SourceData('C:\\Users\\bsolomin\\Astro\\Orion\\Part_three\\cropped\\', non_linear=True),
            # SourceData('C:\\Users\\bsolomin\\Astro\\Orion\\Part_two\\cropped\\', non_linear=True),
            # SourceData('C:\\Users\\bsolomin\\Astro\\M81\\cropped\\', non_linear=True),
            SourceData('C:\\Users\\bsolomin\\Astro\\NGC_1333_RASA\\Fits', non_linear=False),
        ],
        samples_folder='C:\\git\\object_recognition\\star_samples',
    )

    training_generator = dataset.batch_generator(batch_size=10)
    val_generator = dataset.batch_generator(batch_size=10)

    early_stopping_monitor = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=4,
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
            validation_steps=4000,
            epochs=20,
            callbacks=[early_stopping_monitor]
        )
    except KeyboardInterrupt:
        model.save(f"{save_model_name}.h5")
        encrypt_model(save_model_name)

    model.save(f"{save_model_name}.h5")
    encrypt_model(save_model_name)


if __name__ == '__main__':
    main()


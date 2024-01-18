import os

import tensorflow as tf
from backend.source_data import SourceData
from training_dataset import TrainingDataset
from model_builder import build_rnn_model, encrypt_model
from collections import namedtuple
from logger.logger import Logger
logger = Logger()


folder_properties = namedtuple("folder_properties", ("folder", "non_linear", "to_align", "num_from_session"))



def get_file_paths(folder):
    return [os.path.join(folder, item) for item in os.listdir(folder) if item.lower().endswith(".xisf") or item.lower().endswith(".fit") or item.lower().endswith(".fits")]


def main():
    logger.log.info(tf.__version__)
    input_shape = (None, 64, 64, 1)
    load_model_name = "model52"
    save_model_name = "model53"


    # Build the model


    # Compile the model
    # model = build_rnn_model(input_shape)
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Load model
    model = tf.keras.models.load_model(
        f'{load_model_name}.h5'
    )

    folders = [
        # folder_properties('C:\\Users\\bsolomin\\Astro\\SeaHorse\\cropped\\', non_linear=True, to_align=False, num_from_session=None),
        # folder_properties('C:\\Users\\bsolomin\\Astro\\SeaHorse\\cropped\\', non_linear=True, to_align=False, num_from_session=5),
        # folder_properties('C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\cropped', non_linear=True, to_align=False, num_from_session=25),
        # folder_properties('C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\cropped', non_linear=True, to_align=False, num_from_session=None),
        # folder_properties('C:\\Users\\bsolomin\\Astro\\Andromeda\\Pix_600\\cropped\\', non_linear=True, to_align=False, num_from_session=None),
        # folder_properties('C:\\Users\\bsolomin\\Astro\\Andromeda\\Pix_600\\cropped\\', non_linear=True, to_align=False, num_from_session=10),
        # folder_properties('C:\\Users\\bsolomin\\Astro\\Orion\\Part_four\\cropped1\\', non_linear=True, to_align=False, num_from_session=None),
        # folder_properties('C:\\Users\\bsolomin\\Astro\\Orion\\Part_four\\cropped1\\', non_linear=True, to_align=False, num_from_session=5),
        # folder_properties('C:\\Users\\bsolomin\\Astro\\Orion\\Part_one\\cropped\\', non_linear=True, to_align=False, num_from_session=None),
        # folder_properties('C:\\Users\\bsolomin\\Astro\\Orion\\Part_three\\cropped\\', non_linear=True, to_align=False, num_from_session=None),
        # folder_properties('C:\\Users\\bsolomin\\Astro\\Orion\\Part_three\\cropped\\', non_linear=True, to_align=False, num_from_session=5),
        # folder_properties('C:\\Users\\bsolomin\\Astro\\Orion\\Part_two\\cropped\\', non_linear=True, to_align=False, num_from_session=None),
        # folder_properties('C:\\Users\\bsolomin\\Astro\\M81\\cropped\\', non_linear=True, to_align=False, num_from_session=None),
        folder_properties('C:\\Users\\bsolomin\\Astro\\NGC_1333_RASA\\cropped', non_linear=True, to_align=False, num_from_session=5),
        # folder_properties('C:\\Users\\bsolomin\\Astro\\NGC_1333_RASA\\cropped', non_linear=True, to_align=False, num_from_session=None),
        # folder_properties('D:\\Boris\\astro\\Auriga\\Light', non_linear=False, to_align=True, num_from_session=20),
    ]
    source_datas = []
    for item in folders:
        file_paths = get_file_paths(item.folder)
        source_datas.append(SourceData(
            file_paths,
            non_linear=item.non_linear,
            to_align=item.to_align,
            num_from_session=item.num_from_session,
            to_skip_bad=True,
        ))

    dataset = TrainingDataset(source_datas)

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


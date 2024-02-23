import os

import tensorflow as tf
from backend.source_data import SourceData, get_file_paths
from training_dataset import TrainingDataset
from model_builder import build_model, encrypt_model
from collections import namedtuple
from logger.logger import Logger
logger = Logger()


os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


folder_properties = namedtuple(
    "folder_properties",
    ("folder", "non_linear", "to_align", "num_from_session", "debayer", "darks", "flats", "dark_flats"),
    defaults=(True, False, None, False, None, None, None))


def main():
    logger.log.info(tf.__version__)
    input_shape = (None, 64, 64, 1)
    load_model_name = "model112"
    save_model_name = "model113"

    # Build the model

    # Compile the model
    # model = build_model()
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Load model
    model = tf.keras.models.load_model(
        f'{load_model_name}.h5'
    )

    # folders = [
    #     # folder_properties('C:\\Users\\bsolomin\\Astro\\SeaHorse\\cropped\\', non_linear=True, to_align=False, num_from_session=None),
    #     # folder_properties('C:\\Users\\bsolomin\\Astro\\SeaHorse\\cropped\\', non_linear=True, to_align=False, num_from_session=3),
    #     # folder_properties('C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\cropped', non_linear=True, to_align=False, num_from_session=2),
    #     # # folder_properties('C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\cropped', non_linear=True, to_align=False, num_from_session=None),
    #     # # folder_properties('C:\\Users\\bsolomin\\Astro\\Andromeda\\Pix_600\\cropped\\', non_linear=True, to_align=False, num_from_session=None),
    #     # folder_properties('C:\\Users\\bsolomin\\Astro\\Andromeda\\Pix_600\\cropped\\', non_linear=True, to_align=False, num_from_session=10),
    #     # folder_properties('C:\\Users\\bsolomin\\Astro\\Orion\\Part_four\\cropped1\\', non_linear=True, to_align=False, num_from_session=None),
    #     # folder_properties('C:\\Users\\bsolomin\\Astro\\Orion\\Part_four\\cropped1\\', non_linear=True, to_align=False, num_from_session=3),
    #     # folder_properties('C:\\Users\\bsolomin\\Astro\\Orion\\Part_one\\cropped\\', non_linear=True, to_align=False, num_from_session=None),
    #     # folder_properties('C:\\Users\\bsolomin\\Astro\\Orion\\Part_three\\cropped\\', non_linear=True, to_align=False, num_from_session=None),
    #     # folder_properties('C:\\Users\\bsolomin\\Astro\\Orion\\Part_three\\cropped\\', non_linear=True, to_align=False, num_from_session=2),
    #     # folder_properties('C:\\Users\\bsolomin\\Astro\\Orion\\Part_two\\cropped\\', non_linear=True, to_align=False, num_from_session=None),
    #     # folder_properties('C:\\Users\\bsolomin\\Astro\\M81\\cropped\\', non_linear=True, to_align=False, num_from_session=2),
    #     # folder_properties('C:\\Users\\bsolomin\\Astro\\NGC_1333_RASA\\cropped', non_linear=True, to_align=False, num_from_session=5),
    #     # folder_properties('C:\\Users\\bsolomin\\Astro\\NGC_1333_RASA\\cropped', non_linear=True, to_align=False, num_from_session=3),
    #     # folder_properties('C:\\Users\\bsolomin\\Astro\\NGC_1333_RASA\\cropped', non_linear=True, to_align=False, num_from_session=7),
    #     # folder_properties('C:\\Users\\bsolomin\\Astro\\NGC_1333_RASA\\cropped', non_linear=True, to_align=False, num_from_session=None),
    #     # # folder_properties('D:\\Boris\\astro\\Auriga\\Light', non_linear=False, to_align=True, num_from_session=20),
    #     folder_properties('D:\\Boris\\astro\\Veil\\Light\\Part_2', non_linear=False, to_align=True, debayer=True, num_from_session=20),
    # ]
    folders = [
        # folder_properties('D:\\Boris\\astro\\Antares\\Part1\\Night1', non_linear=False, to_align=True,
        #                   num_from_session=None, debayer=True, darks='D:\\Boris\\astro\\Antares\\Dark'),
        # folder_properties('D:\\Boris\\astro\\Antares\\Part2\\Night2', non_linear=False, to_align=True,
        #                   num_from_session=None, debayer=True),
        # folder_properties('C:\\Users\\bsolomin\\Astro\\NGC_1333_RASA\\Light', non_linear=False, to_align=True,
        #                   num_from_session=None, debayer=True, darks='C:\\Users\\bsolomin\\Astro\\NGC_1333_RASA\\Dark'),
        # folder_properties('C:\\Users\\bsolomin\\Astro\\NGC_1333_RASA\\Light', non_linear=False, to_align=True,
        #                   num_from_session=8, debayer=True, darks='C:\\Users\\bsolomin\\Astro\\NGC_1333_RASA\\Dark'),
        # folder_properties('C:\\Users\\bsolomin\\Astro\\NGC_1333_RASA\\Light', non_linear=False, to_align=True,
        #                   num_from_session=6, debayer=True, darks='C:\\Users\\bsolomin\\Astro\\NGC_1333_RASA\\Dark'),
        # folder_properties('C:\\Users\\bsolomin\\Astro\\NGC_1333_RASA\\Light', non_linear=False, to_align=True,
        #                   num_from_session=None, debayer=True),
        # folder_properties('C:\\Users\\bsolomin\\Astro\\Andromeda\\Light\\Night3', non_linear=False, to_align=True,
        #                   num_from_session=None, debayer=True, darks='C:\\Users\\bsolomin\\Astro\\Andromeda\\Dark_600'),
        # folder_properties('D:\\Boris\\astro\\Auriga\\Light', non_linear=False, to_align=True,
        #                   num_from_session=8, debayer=True, darks='D:\\Boris\\astro\\M78\\Dark'),
        # folder_properties('D:\\Boris\\astro\\Auriga\\Light', non_linear=False, to_align=True,
        #                   num_from_session=2, debayer=True, darks='D:\\Boris\\astro\\M78\\Dark'),
        # folder_properties('D:\\Boris\\astro\\NGC6888\\Light', non_linear=False, to_align=True,
        #                   num_from_session=4, debayer=True, darks='D:\\Boris\\astro\\NGC6888\\Dark'),
        # folder_properties('D:\\Boris\\astro\\seahorse\\Light', non_linear=False, to_align=True,
        #                   num_from_session=50, debayer=True, darks='C:\\Users\\bsolomin\\Astro\\SeaHorse\\Dark'),


        folder_properties('C:\\Users\\bsolomin\\Astro\\NGC_1333_RASA\\cropped', non_linear=True, to_align=False,
                          num_from_session=None, debayer=False),
        folder_properties('C:\\Users\\bsolomin\\Astro\\SeaHorse\\cropped', non_linear=True, to_align=False,
                          num_from_session=None, debayer=False),
        folder_properties('C:\\Users\\bsolomin\\Astro\\Orion\\Part_one\\cropped', non_linear=True, to_align=False,
                          num_from_session=None, debayer=False),
        folder_properties('C:\\Users\\bsolomin\\Astro\\Orion\\Part_two\\cropped', non_linear=True, to_align=False,
                          num_from_session=None, debayer=False),
        folder_properties('C:\\Users\\bsolomin\\Astro\\Orion\\Part_three\\cropped', non_linear=True, to_align=False,
                          num_from_session=None, debayer=False),
        folder_properties('C:\\Users\\bsolomin\\Astro\\Orion\\Part_four\\cropped1', non_linear=True, to_align=False,
                          num_from_session=None, debayer=False),
        folder_properties('C:\\Users\\bsolomin\\Astro\\NGC_1333_RASA\\cropped', non_linear=True, to_align=False,
                          num_from_session=5, debayer=False),
        folder_properties('C:\\Users\\bsolomin\\Astro\\SeaHorse\\cropped', non_linear=True, to_align=False,
                          num_from_session=5, debayer=False),
        folder_properties('C:\\Users\\bsolomin\\Astro\\Orion\\Part_one\\cropped', non_linear=True, to_align=False,
                          num_from_session=3, debayer=False),
        folder_properties('C:\\Users\\bsolomin\\Astro\\Orion\\Part_two\\cropped', non_linear=True, to_align=False,
                          num_from_session=6, debayer=False),
        folder_properties('C:\\Users\\bsolomin\\Astro\\Orion\\Part_three\\cropped', non_linear=True, to_align=False,
                          num_from_session=7, debayer=False),
        folder_properties('C:\\Users\\bsolomin\\Astro\\Orion\\Part_four\\cropped1', non_linear=True, to_align=False,
                          num_from_session=4, debayer=False),
        folder_properties('C:\\Users\\bsolomin\\Astro\M78\\registered\\Light_BIN-1_EXPOSURE-120.00s_FILTER-NoFilter_RGB', non_linear=False, to_align=False,
                          num_from_session=None, debayer=False),
        folder_properties('C:\\Users\\bsolomin\\Astro\\M81\\cropped', non_linear=True, to_align=False,
                          num_from_session=15, debayer=False),
        folder_properties('C:\\Users\\bsolomin\\Astro\\Virgo', non_linear=False, to_align=True,
                          num_from_session=None, debayer=True, darks='C:\\Users\\bsolomin\\Astro\\Rosette\\Dark',
                          flats="C:\\Users\\bsolomin\\Astro\\Rosette\\Flat",
                          dark_flats="C:\\Users\\bsolomin\\Astro\\Rosette\\DarkFlat"),
        folder_properties('C:\\Users\\bsolomin\\Astro\\Virgo1', non_linear=False, to_align=True,
                          num_from_session=None, debayer=True, darks='C:\\Users\\bsolomin\\Astro\\Rosette\\Dark',
                          flats="C:\\Users\\bsolomin\\Astro\\Rosette\\Flat",
                          dark_flats="C:\\Users\\bsolomin\\Astro\\Rosette\\DarkFlat"),
        folder_properties('C:\\Users\\bsolomin\\Astro\\Virgo2', non_linear=False, to_align=True,
                          num_from_session=None, debayer=True, darks='C:\\Users\\bsolomin\\Astro\\Rosette\\Dark',
                          flats="C:\\Users\\bsolomin\\Astro\\Rosette\\Flat",
                          dark_flats="C:\\Users\\bsolomin\\Astro\\Rosette\\DarkFlat"),
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
            to_debayer=item.debayer,
            dark_folder=item.darks,
            flat_folder=item.flats,
            dark_flat_folder=item.dark_flats
        ))

    dataset = TrainingDataset(source_datas)

    training_generator = dataset.batch_generator(batch_size=10)
    val_generator = dataset.batch_generator(batch_size=10)

    early_stopping_monitor = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=50,
        verbose=1,
        mode='min',
        baseline=None,
        restore_best_weights=True
    )

    try:
        model.fit(
            training_generator,
            validation_data=val_generator,
            steps_per_epoch=500,
            validation_steps=500,
            epochs=800,
            callbacks=[early_stopping_monitor]
        )
    except KeyboardInterrupt:
        model.save(f"{save_model_name}.h5")
        encrypt_model(save_model_name)

    model.save(f"{save_model_name}.h5")
    encrypt_model(save_model_name)


if __name__ == '__main__':
    main()


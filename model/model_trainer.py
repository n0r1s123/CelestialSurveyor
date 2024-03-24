import os

import tensorflow as tf
from backend.source_data import SourceData, get_file_paths
from training_dataset import TrainingDataset
from model_builder import build_model, encrypt_model
from collections import namedtuple
from logger.logger import Logger
from tensorflow.keras import mixed_precision
logger = Logger()


# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)


folder_properties = namedtuple(
    "folder_properties",
    ("folder", "non_linear", "to_align", "num_from_session", "debayer", "darks", "flats", "dark_flats"),
    defaults=(True, False, None, False, None, None, None))


def main():
    logger.log.info(tf.__version__)
    input_shape = (None, 64, 64, 1)
    # load_model_name = "model129"
    save_model_name = "model131"

    # Build the model

    # Compile the model
    model = build_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Load model
    # model = tf.keras.models.load_model(
    #     f'{load_model_name}.h5'
    # )

    folders = [
        folder_properties('D:\\git\\dataset\\NGC1333_RASA\\cropped', non_linear=True, to_align=False,
                          num_from_session=None, debayer=False),
        folder_properties('D:\\git\\dataset\\Seahorse\\cropped', non_linear=True, to_align=False,
                          num_from_session=None, debayer=False),
        folder_properties('D:\\git\\dataset\\Orion\\Part1\\cropped', non_linear=True, to_align=False,
                          num_from_session=None, debayer=False),
        folder_properties('D:\\git\\dataset\\Orion\Part2\\cropped', non_linear=True, to_align=False,
                          num_from_session=None, debayer=False),
        folder_properties('D:\\git\\dataset\\Orion\Part3\\cropped', non_linear=True, to_align=False,
                          num_from_session=None, debayer=False),
        folder_properties('D:\\git\\dataset\\Orion\\Part4\\cropped1', non_linear=True, to_align=False,
                          num_from_session=None, debayer=False),
        folder_properties('D:\\git\\dataset\\NGC1333_RASA\\cropped', non_linear=True, to_align=False,
                          num_from_session=5, debayer=False),
        folder_properties('D:\\git\\dataset\\Seahorse\\cropped', non_linear=True, to_align=False,
                          num_from_session=5, debayer=False),
        folder_properties('D:\\git\\dataset\\Orion\\Part1\\cropped', non_linear=True, to_align=False,
                          num_from_session=3, debayer=False),
        folder_properties('D:\\git\\dataset\\Orion\Part2\\cropped', non_linear=True, to_align=False,
                          num_from_session=6, debayer=False),
        folder_properties('D:\\git\\dataset\\Orion\Part3\\cropped', non_linear=True, to_align=False,
                          num_from_session=7, debayer=False),
        folder_properties('D:\\git\\dataset\\Orion\\Part4\\cropped1', non_linear=True, to_align=False,
                          num_from_session=4, debayer=False),
        folder_properties('D:\\git\\dataset\\M78\\Light_BIN-1_EXPOSURE-120.00s_FILTER-NoFilter_RGB', non_linear=False, to_align=False,
                          num_from_session=None, debayer=False),
        folder_properties('D:\\git\\dataset\\M81\\cropped', non_linear=True, to_align=False,
                          num_from_session=15, debayer=False),
        # folder_properties('D:\\git\\dataset\Virgo', non_linear=False, to_align=True,
        #                   num_from_session=None, debayer=True, darks='D:\\git\\dataset\\Virgo\\Dark',
        #                   flats="D:\\git\\dataset\\Virgo\\Flat",
        #                   dark_flats="D:\\git\\dataset\\Virgo\\DarkFlat"),
        # folder_properties('D:\\git\\dataset\Virgo1', non_linear=False, to_align=True,
        #                   num_from_session=None, debayer=True, darks='D:\\git\\dataset\\Virgo\\Dark',
        #                   flats="D:\\git\\dataset\\Virgo\\Flat",
        #                   dark_flats="D:\\git\\dataset\\Virgo\\DarkFlat"),
        # folder_properties('D:\\git\\dataset\Virgo2', non_linear=False, to_align=True,
        #                   num_from_session=None, debayer=True, darks='D:\\git\\dataset\\Virgo\\Dark',
        #                   flats="D:\\git\\dataset\\Virgo\\Flat",
        #                   dark_flats="D:\\git\\dataset\\Virgo\\DarkFlat"),
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

    training_generator = dataset.batch_generator(batch_size=40)
    val_generator = dataset.batch_generator(batch_size=40)

    early_stopping_monitor = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=100,
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
            validation_steps=250,
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


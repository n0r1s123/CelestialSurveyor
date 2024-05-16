
# if __name__ == "__main__":
import os

import tensorflow as tf
from training_dataset_v2 import TrainingDatasetV2, TrainingSourceDataV2
from model_builder import build_model, encrypt_model
from collections import namedtuple
from logger.logger import Logger
from dataclasses import dataclass
from typing import Optional
from backend.progress_bar import ProgressBarCli
from tensorflow.keras import mixed_precision
logger = Logger()


# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)


@dataclass
class SourceDataProperties:
    folder: str
    linear: bool
    to_align: bool
    to_debayer: bool
    secondary_alignment: tuple[int, int] = (3, 3)
    number_of_images: Optional[int] = None
    dark_folder: Optional[str] = None
    flat_folder: Optional[str] = None
    dark_flats_folder: Optional[str] = None

    @property
    def file_paths(self):
        return SourceDataProperties.make_file_paths(self.folder)

    @property
    def dark_paths(self):
        if self.dark_folder is None:
            return None
        return SourceDataProperties.make_file_paths(self.dark_folder)

    @property
    def flat_paths(self):
        if self.flat_folder is None:
            return None
        return SourceDataProperties.make_file_paths(self.flat_folder)

    @property
    def dark_flat_paths(self):
        if self.dark_flats_folder is None:
            return None
        return SourceDataProperties.make_file_paths(self.dark_flats_folder)


def main():
    logger.log.info(tf.__version__)
    input_shape = (None, 64, 64, 1)
    load_model_name = "model131"
    save_model_name = "model132"

    # Build the model

    # Compile the model
    # model = build_model()
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Load model
    model = tf.keras.models.load_model(
        f'{load_model_name}.h5'
    )

    source_data_properties = [
        SourceDataProperties(
            folder='D:\\git\\dataset\\NGC1333_RASA\\cropped',
            linear=False,
            to_align=False,
            number_of_images=None,
            to_debayer=False),
        SourceDataProperties(
            folder='D:\\git\\dataset\\Seahorse\\cropped',
            linear=False,
            to_align=False,
            number_of_images=None,
            to_debayer=False),
        SourceDataProperties(
            folder='D:\\git\\dataset\\Orion\\Part1\\cropped',
            linear=False,
            to_align=False,
            number_of_images=None,
            to_debayer=False),
        SourceDataProperties(
            folder='D:\\git\\dataset\\Orion\\Part2\\cropped',
            linear=False,
            to_align=False,
            number_of_images=None,
            to_debayer=False),
        SourceDataProperties(
            folder='D:\\git\\dataset\\Orion\\Part3\\cropped',
            linear=False,
            to_align=False,
            number_of_images=None,
            to_debayer=False),
        SourceDataProperties(
            folder='D:\\git\\dataset\\Orion\\Part4\\cropped1',
            linear=False,
            to_align=False,
            number_of_images=None,
            to_debayer=False),
        SourceDataProperties(
            folder='D:\\git\\dataset\\NGC1333_RASA\\cropped',
            linear=False,
            to_align=False,
            number_of_images=5,
            to_debayer=False),
        SourceDataProperties(
            folder='D:\\git\\dataset\\Seahorse\\cropped',
            linear=False,
            to_align=False,
            number_of_images=7,
            to_debayer=False),
        SourceDataProperties(
            folder='D:\\git\\dataset\\Orion\\Part1\\cropped',
            linear=False,
            to_align=False,
            number_of_images=3,
            to_debayer=False),
        SourceDataProperties(
            folder='D:\\git\\dataset\\Orion\\Part2\\cropped',
            linear=False,
            to_align=False,
            number_of_images=6,
            to_debayer=False),
        SourceDataProperties(
            folder='D:\\git\\dataset\\Orion\\Part3\\cropped',
            linear=False,
            to_align=False,
            number_of_images=7,
            to_debayer=False),
        SourceDataProperties(
            folder='D:\\git\\dataset\\Orion\\Part4\\cropped1',
            linear=False,
            to_align=False,
            number_of_images=4,
            to_debayer=False),
        SourceDataProperties(
            folder='D:\\git\\dataset\\M78\\Light_BIN-1_EXPOSURE-120.00s_FILTER-NoFilter_RGB',
            linear=True,
            to_align=False,
            number_of_images=None,
            to_debayer=False),
        SourceDataProperties(
            folder='D:\\git\\dataset\\M81\\cropped',
            linear=False,
            to_align=False,
            number_of_images=60,
            to_debayer=False),
        SourceDataProperties(
            folder='D:\\git\\dataset\\Virgo',
            linear=True,
            to_align=True,
            number_of_images=None,
            to_debayer=True,
            dark_folder='D:\\git\\dataset\\Virgo\\Dark',
            flat_folder='D:\\git\\dataset\\Virgo\\Flat',
            dark_flats_folder='D:\\git\\dataset\\Virgo\\DarkFlat'),
        SourceDataProperties(
            folder='D:\\git\\dataset\\Virgo1',
            linear=True,
            to_align=True,
            number_of_images=None,
            to_debayer=True,
            dark_folder='D:\\git\\dataset\\Virgo\\Dark',
            flat_folder='D:\\git\\dataset\\Virgo\\Flat',
            dark_flats_folder='D:\\git\\dataset\\Virgo\\DarkFlat'),
        SourceDataProperties(
            folder='D:\\git\\dataset\\Virgo2',
            linear=True,
            to_align=True,
            number_of_images=None,
            to_debayer=True,
            dark_folder='D:\\git\\dataset\\Virgo\\Dark',
            flat_folder='D:\\git\\dataset\\Virgo\\Flat',
            dark_flats_folder='D:\\git\\dataset\\Virgo\\DarkFlat'),
    ]

    source_datas = []
    for num, properties in enumerate(source_data_properties):
        logger.log.info(f"Processing {num+1} of {len(source_data_properties)} source datas")
        source_data = TrainingSourceDataV2(properties.file_paths, properties.to_debayer, properties.number_of_images)
        source_data.load_images(progress_bar=ProgressBarCli())
        logger.log.info(f"dtype = {source_data.original_frames.dtype}")
        source_data.calibrate_images(properties.dark_paths, properties.flat_paths, properties.dark_flat_paths,
                                     progress_bar=ProgressBarCli())
        if properties.to_align:
            source_data.align_images(progress_bar=ProgressBarCli())

        # TODO: Crop images if alignment is not required
        source_data.crop_images()
        if properties.to_align:
            source_data.secondary_align_images(*properties.secondary_alignment, progress_bar=ProgressBarCli())

        if properties.linear:
            source_data.stretch_images(progress_bar=ProgressBarCli())
        source_data.load_exclusion_boxes()
        source_data.images_from_buffer()
        source_datas.append(source_data)

        size = source_data.images.itemsize
        for item in source_data.images.shape:
            size *= item
        logger.log.info(f"dtype: {source_data.images.dtype}, shape: {source_data.images.shape}, size: {size}")
        logger.log.info(f"Allocated {size // (1024 * 1024)} Mb")

    training_dataset = TrainingDatasetV2(source_datas)
    training_generator = training_dataset.batch_generator(batch_size=20)
    val_generator = training_dataset.batch_generator(batch_size=20)

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


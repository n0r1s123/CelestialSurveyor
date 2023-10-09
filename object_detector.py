import random
import time

import cv2
import os
import datetime
import matplotlib.pyplot as plt
from scipy import ndimage

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, Flatten, Conv2D, MaxPooling2D
from dataset_creator.dataset_creator import DatasetCreator
from xisf import XISF

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# Assuming you have a function to generate batches of data
class Dataset:
    def __init__(self, folder, samples_folder):
        self.raw_dataset, self.exposures, self.timestamps, self.img_shape = self.__load_raw_dataset(folder)
        self.object_samples = self.__load_samples(samples_folder)
        self.example_generated = False


    def __load_raw_dataset(self, folder):
        file_list = [os.path.join(folder, item) for item in os.listdir(folder)]
        raw_dataset = np.array([np.array(XISF(item).read_image(0)[:, :, 0]) for item in file_list])
        timestamps = []
        exposures = []
        for num, item in enumerate(file_list, start=1):
            img_meta = XISF(item).get_images_metadata()[0]
            exposure = float(img_meta["FITSKeywords"]["EXPTIME"][0]['value'])
            timestamp = img_meta["FITSKeywords"]["DATE-OBS"][0]['value']
            timestamp = datetime.datetime.strptime(timestamp.replace("T", " "), '%Y-%m-%d %H:%M:%S.%f')
            exposures.append(exposure)
            timestamps.append(timestamp)

        print("Raw image dataset loaded:")
        print(f"LEN: {len(raw_dataset)}")
        print(f"SHAPE: {raw_dataset.shape}")
        print(f"Memory: {(raw_dataset.itemsize * raw_dataset.size) // (1024 * 1024)} Mb")
        print(f"Timestamps: {len(timestamps)}")
        img_shape = raw_dataset.shape[1:]

        return raw_dataset, exposures, timestamps, img_shape

    def __load_samples(self, folder):
        file_list = [os.path.join(folder, item) for item in os.listdir(folder)]
        samples = np.array([np.array(XISF(item).read_image(0)[:, :, 0]) for item in file_list])
        return samples

    def get_shrinked_img_series(self, size, y, x):
        shrinked = np.array([cv2.resize(item[y:y+size, x:x+size], dsize=(56, 56), interpolation=cv2.INTER_CUBIC) for item in self.raw_dataset])
        return shrinked

    def get_random_shrink(self):
        size = random.randint(56, min(self.img_shape[:2]))
        y = random.randint(0, self.img_shape[0] - size)
        x = random.randint(0, self.img_shape[1] - size)
        return size, y, x

    @classmethod
    def insert_star_by_coords(cls, image, star, coords):
        """

        :param image: filepath or 2 axis np.array where the star will be drawn.
        :param star: filepath or 2 axis np.array which represents star to be drawn.
        :param coords: tuple with (y, x) coordinates where the star will be drawn on the image.
        :return: 2 axis np.array.
        """
        star_y_size, star_x_size = star.shape[:2]

        image_y_size, image_x_size = image.shape
        x, y = coords
        x = round(x)
        y = round(y)
        if x + star_x_size // 2 < 0 or x - star_x_size // 2 > image_x_size:
            return image
        if y + star_y_size // 2 < 0 or y - star_y_size // 2 > image_y_size:
            return image

        cut_top = y - star_y_size // 2
        cut_top = -cut_top if cut_top < 0 else 0
        cut_bottom = image_y_size - y - star_y_size // 2
        cut_bottom = -cut_bottom if cut_bottom < 0 else 0
        cut_left = x - star_x_size // 2
        cut_left = -cut_left if cut_left < 0 else 0
        cut_right = image_x_size - x - star_x_size // 2
        cut_right = -cut_right if cut_right < 0 else 0

        y_slice = slice(y + cut_top - star_y_size // 2, y - cut_bottom + star_y_size // 2)
        x_slice = slice(x + cut_left - star_x_size // 2, x - cut_right + star_x_size // 2)


        image_to_correct = image[y_slice, x_slice]

        image[y_slice, x_slice] = np.maximum(
            star[int(cut_top):int(star_y_size - cut_bottom), int(cut_left):int(star_x_size - cut_right)], image_to_correct)
        return image

    def calculate_star_form_on_single_image(cls, image, star, start_coords, movement_vector, exposure_time=None):
        """
        Draws star on the image keeping in mind that it may move due to exposure time.
        :param image: filepath or 2 axis np.array where the star will be drawn.
        :param star: filepath or 2 axis np.array which represents star to be drawn.
        :param start_coords: tuple with (y, x) coordinates from where the star starts moving.
        :param movement_vector: tuple with (vy, vx) movements per axis in pixels per hour.
        :param exposure_time: exposure time should be provided in case if image provided as np.ndarray. in case of xisf
                              file it will be taken from image metadata.
        :return: 2 axis np.array.
        """
        per_image_movement_vector = np.array(movement_vector) * exposure_time / 3600
        y_move, x_move = per_image_movement_vector
        start_y, start_x = start_coords
        x_moves_per_y_moves = x_move / y_move
        dx = 0
        for dy in range(round(y_move + 1)):
            if dy * x_moves_per_y_moves // 1 > dx:
                dx += 1
            image = cls.insert_star_by_coords(image, star, (start_y + dy, start_x + dx))
        return image

    def draw_object_on_image_series_numpy(self, imgs):
        result = []
        star_img = random.choice(self.object_samples)
        start_image_idx = random.randint(0, len(imgs) - 1)
        start_y = random.randint(0, 56 - 1)
        start_x = random.randint(0, 56 - 1)
        movement_vector = np.array([random.randint(3, 10), random.randint(3, 10)])
        try:
            start_ts = self.timestamps[start_image_idx]
        except IndexError:
            print(start_image_idx)
            raise


        movement_vector = - movement_vector
        to_beginning_slice = slice(None, start_image_idx)
        # y, x = start_y, start_x
        for img, exposure, timestamp in zip(imgs[to_beginning_slice][::-1], self.exposures[to_beginning_slice], self.timestamps[to_beginning_slice]):
            inter_image_movement_vector = np.array(movement_vector) * (timestamp - start_ts).total_seconds() / 3600
            y, x = inter_image_movement_vector + np.array([start_y, start_x])
            new_img = self.calculate_star_form_on_single_image(img, star_img, (y, x), movement_vector, exposure)
            result.append(new_img)

        result = result[::-1]

        movement_vector = - movement_vector

        to_end_slice = slice(start_image_idx, None, None)
        for img, exposure, timestamp in zip(imgs[to_end_slice], self.exposures[to_end_slice], self.timestamps[to_end_slice]):
            inter_image_movement_vector = np.array(movement_vector) * (timestamp - start_ts).total_seconds() / 3600
            y, x = inter_image_movement_vector + np.array([start_y, start_x])
            new_img = self.calculate_star_form_on_single_image(img, star_img, (y, x), movement_vector, exposure)
            result.append(new_img)
        result = np.array(result)
        return result

    def generate_series(self):
        while True:
            shrinked = self.get_shrinked_img_series(*self.get_random_shrink())
            if random.randint(0, 1):
                imgs = self.draw_object_on_image_series_numpy(shrinked)
                imgs = np.array([imgs[num] - imgs[0] for num in range(1, len(imgs))])
                imgs[imgs < 0] = 0
                imgs = np.array([(data - np.min(data))/(np.max(data) - np.min(data)) for data in imgs])
                imgs = imgs ** 2

                imgs.shape = *imgs.shape, 1
                result = imgs, np.array([1])
                if not self.example_generated:
                    for num, item in enumerate(imgs):
                        XISF.write(
                            os.path.join('C:\\git\\object_recognition\\examples', f"{num:03}.xisf"), item,
                            creator_app="My script v1.0",
                            codec='lz4hc', shuffle=True
                        )
                    self.example_generated = True
            else:
                shrinked = np.array([shrinked[num] - shrinked[0] for num in range(1, len(shrinked))])
                shrinked[shrinked < 0] = 0
                shrinked = np.array([(data - np.min(data)) / (np.max(data) - np.min(data)) for data in shrinked])
                shrinked = shrinked ** 2
                shrinked.shape = *shrinked.shape, 1
                result = shrinked, np.array([0])
            yield result

    def generate_batch(self, batch_size):
        series_generator = self.generate_series()
        while True:
            timestamps_batch = []
            batch = [next(series_generator) for _ in range(batch_size)]
            X_batch = np.array([item[0] for item in batch])
            y_batch = np.array([item[1] for item in batch])
            normalizer = (max(self.timestamps) - min(self.timestamps)).total_seconds()
            timestamps = [(max(self.timestamps) - ts).total_seconds() / normalizer for ts in self.timestamps]
            for _ in range(batch_size):
                timestamps_batch.append(timestamps)
            yield X_batch, y_batch


# Define your RNN model
def build_rnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
        #
        # # tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu')),
        # # tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
        # #
        # # tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, (3, 3), activation='relu')),
        # # tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),

        #
        # tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128), input_shape=input_shape),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten(), input_shape=input_shape),

        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(64, return_sequences=False),

        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    dataset = Dataset('C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\cropped',
                                        'C:\\git\\object_recognition\\star_samples')
    def __init__(self, list_IDs=None, labels=None, batch_size=32, n_channels=1,
                 n_classes=1, shuffle=False):
        'Initialization'
        # self.dataset = Dataset('C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\cropped',
        #                                 'C:\\git\\object_recognition\\star_samples')
        self.dim = (len(self.dataset.raw_dataset), 56, 56, 1)
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.generator = self.dataset.generate_batch(self.batch_size)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs) // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        X, y = next(self.generator)
        y.shape = (self.batch_size, 1)
        return X, y


if __name__ == '__main__':
    print(tf.__version__)
    input_shape = (None, 56, 56, 1)
    # Build the model

    # # Compile the model
    # model = build_rnn_model(input_shape)
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.build()


    # Load model
    model = tf.keras.models.load_model(
        'model3.h5'
    )

    print(model.summary())

    training_generator = DataGenerator(range(10000), range(10000), batch_size=10)
    val_generator = DataGenerator(range(1000), range(1000), batch_size=10)
    try:
        model.fit_generator(generator=training_generator,
                            validation_data=val_generator,
                            epochs=5,
                            )
    except KeyboardInterrupt:
        model.save("model4.h5")
    model.save("model4.h5")


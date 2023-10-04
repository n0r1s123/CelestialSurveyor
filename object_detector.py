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

# Assuming you have a function to generate batches of data
class Dataset:
    def __init__(self, folder, samples_folder):
        self.raw_dataset, self.exposures, self.timestamps, self.img_shape = self.__load_raw_dataset(folder)
        self.object_samples = self.__load_samples(samples_folder)


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
        shrinked = np.array([cv2.resize(item[y:y+size, x:x+size], dsize=(224, 224), interpolation=cv2.INTER_CUBIC) for item in self.raw_dataset])
        return shrinked

    def get_random_shrink(self):
        size = random.randint(224, min(self.img_shape[:2]))
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
        start_y = random.randint(0, 253)
        start_x = random.randint(0, 253)
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
        # result.shape = *result.shape, 1
        # print(f"RESULT SHAPE: {result.shape}")
        return result

    def generate_series(self):
        while True:
            shrinked = self.get_shrinked_img_series(*self.get_random_shrink())
            if random.randint(0, 1):
                imgs = self.draw_object_on_image_series_numpy(shrinked)
                imgs = np.array([imgs[num] - imgs[0] for num in range(1, len(imgs))])
                imgs.shape = *imgs.shape, 1
                result = imgs, 1
            else:
                shrinked.shape = *shrinked.shape, 1
                shrinked = np.array([shrinked[num] - shrinked[0] for num in range(1, len(shrinked))])
                result = shrinked, 0
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





# def data_batch_generator(batch_size):
#     while True:
#         # Generate a batch of data (X, timestamps, y)
#         # For example, you might load a batch of images, timestamps, and labels
#         # Make sure to handle data augmentation, preprocessing, etc. here
#
#         yield {'images': X_batch, 'timestamps': timestamps_batch}, y_batch
#

# Define your RNN model
def build_rnn_model(input_shape):
    # model = tf.keras.Sequential([
    #     tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape),
    #     tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
    #     tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
    #     tf.keras.layers.SimpleRNN(64, activation='relu'),
    #     tf.keras.layers.Dense(1, activation='sigmoid')
    # ])
    model = tf.keras.Sequential([
        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),

        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu')),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),

        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, (3, 3), activation='relu')),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),

        tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
        tf.keras.layers.SimpleRNN(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


if __name__ == '__main__':
    # dataset_class = Dataset('C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\cropped',
    #                                     'C:\\git\\object_recognition\\star_samples')

    # gen = dataset_class.generate_series()
    # result = 1
    # series = None
    # while result:
    #     series, result = next(gen)
    # print(len(series))
    # # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (224, 224), False)
    # for item in series:
    #     med_denoised = ndimage.gaussian_filter(item, 2)
    #     plt.imshow(med_denoised, cmap='gray')
    #     plt.show()


    input_shape = (None, 224, 224, 1)
    # Build the model
    model = build_rnn_model(input_shape)
    print(model.summary())

    # Create a Dataset using from_generator
    num_samples = 10
    # input_shape = (None, 224, 224, 1)  # Variable time steps, 224x224 images, 1 channel (monochrome)
    dataset_class = Dataset('C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\cropped',
                                        'C:\\git\\object_recognition\\star_samples')

    dataset = tf.data.Dataset.from_generator(
        generator=dataset_class.generate_series,
        output_types=(tf.float32, tf.int32),  # Adjust types if necessary
        output_shapes=(input_shape, ())
    )
    # Batch and prefetch the dataset
    batch_size = 1
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(dataset, epochs=5, steps_per_epoch=200)



# ============================================================


    # # Define the model
    # model = Sequential([
    #     Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224)),
    #     MaxPooling2D(pool_size=(2, 2)),
    #     Flatten(),
    #     LSTM(64, return_sequences=True),
    #     LSTM(32),
    #     Dense(1, activation='sigmoid')
    # ])
    #
    # # Compile the model
    # model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    # print(model.summary())
    #
    # # Create a dataset using your custom generator
    # batch_size = 32
    # dataset = Dataset('C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\cropped',
    #                   'C:\\git\\object_recognition\\star_samples')
    # train_dataset = tf.data.Dataset.from_generator(dataset.generate_batch, args=[batch_size],
    #                                                output_types=(tf.float32, tf.float32))
    # train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    #
    # # Train the model using the dataset
    # epochs = 10
    # steps_per_epoch = 10
    # model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch)
    #
    # # # Evaluate the model (similarly create a test_dataset if needed)
    # # loss, accuracy = model.evaluate(test_dataset)
    # #







# if __name__ == '__main__':
#     dataset = Dataset('C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\cropped', 'C:\\git\\object_recognition\\star_samples')
#     generator = dataset.generate_series()
#     print(generator)
#     # shrinked = dataset.get_shrinked_img_series(*dataset.get_random_shrink())
#     start_time = time.time()
#     result, idx, x, y = next(generator)
#     print(f"It took {time.time() - start_time} seconds to generate")
#     # print(len(shrinked))
#     print(x, y)
#     print(len(result))
#     for num in range(max([idx - 10, 0]), idx + 10):
#         plt.imshow(result[num], cmap='gray')
#         plt.show()

import datetime
import os
import random

import cv2
import numpy as np
import pathlib
import tensorflow as tf

import matplotlib.pyplot as plt


from auto_stretch.stretch import Stretch
from xisf import XISF
from PIL import Image


class SourceData:
    def __init__(self, folder=None, samples_folder=None):
        self.raw_dataset, self.exposures, self.timestamps, self.img_shape = self.__load_raw_dataset(folder)
        normalized_timestamps = [(item - min(self.timestamps)).total_seconds() for item in self.timestamps]
        self.normalized_timestamps = np.array(
            [item / max(normalized_timestamps) for item in normalized_timestamps])
        self.object_samples = self.__load_samples(samples_folder)
        self.max_value = np.max(self.raw_dataset)
        self.min_value = np.min(self.raw_dataset)
        self.average_value = np.average(self.raw_dataset)

    @classmethod
    def __load_raw_dataset(cls, folder):
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

    @classmethod
    def __load_samples(cls, samples_folder):
        file_list = [os.path.join(samples_folder, item) for item in os.listdir(samples_folder) if ".tif" in item]
        samples = np.array([np.array(Image.open(item)) for item in file_list])
        # print(len(samples))
        # for item in samples:
        #     print(item)
        #     plt.imshow(item, cmap='gray')
        #     plt.show()
        # raise Exception
        return samples


class Dataset:
    ZERO_TOLERANCE = 100

    def __init__(self, source_data: SourceData):
        self.source_data = source_data
        # self.raw_dataset = source_data.raw_dataset
        # self.exposures = source_data.exposures
        # self.timestamps = source_data.timestamps
        # self.img_shape = source_data.img_shape
        # self.object_samples = source_data.object_samples
        # self.average_value = source_data.average_value
        # self.max_value = source_data.max_value
        # self.example_generated = False

    @classmethod
    def stretch_image(cls, img_data):
        return Stretch().stretch(img_data)

    @classmethod
    def to_gray(cls, img_data):
        return np.dot(img_data[..., :3], [0.2989, 0.5870, 0.1140])

    @classmethod
    def crop_image(cls, img_data, y_borders, x_borders):
        x_left, x_right = x_borders
        y_top, y_bottom = y_borders
        return img_data[y_top: y_bottom, x_left: x_right]

    @classmethod
    def crop_raw(cls, img_data, to_do=True):
        y_top = x_left = 0
        y_bottom, x_right = img_data.shape
        for num, line in enumerate(img_data):
            if line[-1] != 0 or line[0] != 0:
                y_top = num
                break
        for num, line in enumerate(img_data[::-1]):
            if line[-1] != 0 or line[0] != 0:
                y_bottom -= num
                break

        for num, line in enumerate(img_data.T):
            if line[-1] != 0 or line[0] != 0:
                x_left = num
                break

        for num, line in enumerate(img_data.T[::-1]):
            if line[-1] != 0 or line[0] != 0:
                x_right -= num
                break
        if to_do:
            return cls.crop_image(img_data, (y_top, y_bottom), (x_left, x_right))
        else:
            return (y_top, y_bottom), (x_left, x_right)

    @classmethod
    def crop_fine(cls, img_data, x_pre_crop_boarders=None, y_pre_crop_boarders=None, to_do=True):
        if x_pre_crop_boarders is None:
            _, x_pre_crop_boarders = img_data.shape
        if y_pre_crop_boarders is None:
            y_pre_crop_boarders, _ = img_data.shape

        pre_cropped = img_data[slice(*y_pre_crop_boarders), slice(*x_pre_crop_boarders)]
        y_top_zeros = np.count_nonzero(pre_cropped[0] == 0)
        y_bottom_zeros = np.count_nonzero(pre_cropped[-1] == 0)
        x_left_zeros = np.count_nonzero(pre_cropped.T[0] == 0)
        x_right_zeros = np.count_nonzero(pre_cropped.T[-1] == 0)
        zeros = y_top_zeros, y_bottom_zeros, x_left_zeros, x_right_zeros
        trim_args = (1, False), (-1, False), (1, True), (-1, True)
        args_order = (item[1] for item in sorted(zip(zeros, trim_args), key=lambda x: x[0], reverse=True))

        def _fine_crop_border(img_data_tmp, direction, transpon=True):
            if transpon:
                img_data_tmp = img_data_tmp.T
            x = 0
            for num, line in enumerate(img_data_tmp[::direction]):
                if np.count_nonzero(line == 0) <= np.count_nonzero(
                        img_data_tmp[::direction][num + 1] == 0) and np.count_nonzero(line == 0) < cls.ZERO_TOLERANCE:
                    x = num
                    break
            if direction == -1:
                result_tmp = img_data_tmp[: (x + 1) * direction]
                x = img_data_tmp.shape[0] - x
            else:
                result_tmp = img_data_tmp[x:]
            return result_tmp.T if transpon else result_tmp, x

        cropped = pre_cropped
        if to_do:
            for pair in args_order:
                cropped, _ = _fine_crop_border(cropped, *pair)
            return cropped
        else:
            border_map = {item: value for item, value in zip(trim_args, ["y_top", "y_bottom", "x_left", "x_right"])}
            result = {}
            for pair in args_order:
                boarder_name = border_map[pair]
                cropped, x = _fine_crop_border(cropped, *pair)
                result.update({boarder_name: x})
            return np.array([[result["y_top"], result["y_bottom"]], [result["x_left"], result["x_right"]]]
                            ) + np.array([(y_pre_crop_boarders[0], y_pre_crop_boarders[0]),
                                          (x_pre_crop_boarders[0], x_pre_crop_boarders[0])])

    @classmethod
    def crop_folder(cls, input_folder, output_folder):
        if not os.path.exists(output_folder):
            pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
        file_list = [item for item in os.listdir(input_folder) if "c_d_r.xisf" in item]
        timestamped_file_list = []
        for item in file_list:
            fp = os.path.join(input_folder, item)
            xisf = XISF(fp)
            timestamp = xisf.get_images_metadata()[0]["FITSKeywords"]["DATE-OBS"][0]['value']
            timestamp = datetime.datetime.strptime(timestamp.replace("T", " "), '%Y-%m-%d %H:%M:%S.%f')
            timestamped_file_list.append((fp, timestamp))
        timestamped_file_list.sort(key=lambda x: x[1])
        boarders = np.array([])
        for fp, timestamp in timestamped_file_list:
            xisf = XISF(fp)
            img_data = xisf.read_image(0)
            img_data = Dataset.to_gray(img_data)
            y_boarders, x_boarders = Dataset.crop_raw(img_data, to_do=False)
            y_boarders, x_boarders = Dataset.crop_fine(
                img_data, y_pre_crop_boarders=y_boarders, x_pre_crop_boarders=x_boarders, to_do=False)
            boarders = np.append(boarders, np.array([*y_boarders, *x_boarders]))
        y_boarders = int(np.max(boarders[::4])), int(np.min(boarders[1::4]))
        x_boarders = int(np.max(boarders[2::4])), int(np.min(boarders[3::4]))
        for num, (fp, timestamp) in enumerate(timestamped_file_list, start=1):
            print(f"{num} Cropping image {fp}")
            xisf = XISF(fp)
            file_meta = xisf.get_file_metadata()
            img_meta = xisf.get_images_metadata()
            img_data = xisf.read_image(0)
            img_data = Dataset.to_gray(img_data)
            img_data = Dataset.crop_image(img_data, y_boarders, x_boarders)
            img_data = Dataset.stretch_image(img_data)
            img_data = np.array(img_data)
            img_data = np.float32(img_data)
            img_data.shape = *img_data.shape, 1
            XISF.write(
                os.path.join(output_folder, f"image_{num:04}.xisf"), img_data,
                creator_app="My script v1.0", image_metadata=img_meta[0], xisf_metadata=file_meta,
                codec='lz4hc', shuffle=True
            )

    def get_shrinked_img_series(self, size, y, x, dataset=None):
        dataset = self.source_data.raw_dataset if dataset is None else dataset
        shrinked = np.copy(dataset[:, y:y+size, x:x+size])
        # shrinked = np.array([cv2.resize(item[y:y+size, x:x+size], dsize=(54, 54), interpolation=cv2.INTER_CUBIC) for item in dataset])
        return shrinked

    def get_random_shrink(self):
        size = 54
        # size = random.randint(54, min(self.img_shape[:2]))
        y = random.randint(0, self.source_data.img_shape[0] - size)
        x = random.randint(0, self.source_data.img_shape[1] - size)
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
            elif dy * x_moves_per_y_moves // 1 < -dx:
                dx -= 1
            image = cls.insert_star_by_coords(image, star, (start_y + dy, start_x + dx))
        return image

    def draw_object_on_image_series_numpy(self, imgs):
        # print(imgs.shape)
        average = np.average(imgs)
        result = []
        y_shape, x_shape = imgs[0].shape[:2]
        star_img = random.choice(self.source_data.object_samples)
        start_image_idx = random.randint(0, len(imgs) - 1)
        # start_image_idx = random.randint(0, 1)
        start_y = random.randint(0, y_shape - 1)
        start_x = random.randint(0, x_shape - 1)
        # object_factor = random.choice((0.2,))
        # object_factor = random.choice((0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1))
        object_factor = random.choice((0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1))
        star_max = np.max(star_img)
        # expected_max = self.source_data.average_value + (self.source_data.max_value - self.source_data.average_value) * object_factor
        expected_max = np.average(imgs) + (np.max(imgs) - np.average(imgs)) * object_factor
        multiplier = expected_max / star_max
        star_img = star_img * multiplier

        # print('=' * 60)
        # print(start_y, start_x)
        # print('=' * 60)
        max_vector = 300
        min_vector = -300
        divider = 10
        choices = list(range(min_vector, 0)) + list(range(1, max_vector))
        movement_vector = np.array([random.choice(choices)/divider, random.choice(choices)/divider])
        start_ts = self.source_data.timestamps[start_image_idx]

        movement_vector = - movement_vector
        to_beginning_slice = slice(None, start_image_idx)
        for img, exposure, timestamp in zip(
                imgs[to_beginning_slice][::-1], self.source_data.exposures[to_beginning_slice], self.source_data.timestamps[to_beginning_slice]
        ):
            inter_image_movement_vector = np.array(movement_vector) * (timestamp - start_ts).total_seconds() / 3600
            y, x = inter_image_movement_vector + np.array([start_y, start_x])
            new_img = self.calculate_star_form_on_single_image(img, star_img, (y, x), movement_vector, exposure)
            result.append(new_img)

        result = result[::-1]

        movement_vector = - movement_vector

        to_end_slice = slice(start_image_idx, None, None)
        for img, exposure, timestamp in zip(
                imgs[to_end_slice], self.source_data.exposures[to_end_slice], self.source_data.timestamps[to_end_slice]
        ):
            inter_image_movement_vector = np.array(movement_vector) * (timestamp - start_ts).total_seconds() / 3600
            y, x = inter_image_movement_vector + np.array([start_y, start_x])
            new_img = self.calculate_star_form_on_single_image(img, star_img, (y, x), movement_vector, exposure)
            result.append(new_img)
        result = np.array(result)
        # new_average = np.average(result)
        #
        # if new_average - average == 0:
        #     drawn = 0
        # else:
        #     drawn = 1
        return result, None

    @classmethod
    def prepare_images(cls, imgs):
        # imgs = np.array(
        #     [np.amax(np.array([imgs[num] - imgs[0], imgs[num] - imgs[-1]]), axis=0) for num in range(1, len(imgs) - 1)])
        imgs = np.array([item - imgs[0] for item in imgs[1:-1]])
        # imgs = (imgs - np.min(imgs)) / (np.max(imgs) - np.min(imgs))
        # imgs = imgs - np.average(imgs)
        imgs[imgs < 0] = 0
        imgs = (imgs - np.min(imgs)) / (np.max(imgs) - np.min(imgs))
        # imgs = imgs ** 2
        imgs.shape = (*imgs.shape, 1)
        return imgs

    def make_series(self):
        shrinked = self.get_shrinked_img_series(*self.get_random_shrink())
        if random.randint(0, 100) > 50:
            imgs, drawn = self.draw_object_on_image_series_numpy(shrinked)
            imgs = self.prepare_images(imgs)
            result = imgs, np.array([1])
        else:
            shrinked = self.prepare_images(shrinked)
            result = shrinked, np.array([0])
        return result

    def make_batch(self, batch_size):
        batch = [self.make_series() for _ in range(batch_size)]
        X_batch = np.array([item[0] for item in batch])
        TS_batch = np.array([self.source_data.normalized_timestamps[1: -1] for _ in batch])
        y_batch = np.array([item[1] for item in batch])
        # print(X_batch.shape)
        return X_batch, y_batch

    def batch_generator(self, batch_size):
        while True:
            yield self.make_batch(batch_size)

    # def generate_series(self):
    #     while True:
    #         shrinked = self.get_shrinked_img_series(*self.get_random_shrink())
    #         if random.randint(0, 100) > 90:
    #             imgs = self.draw_object_on_image_series_numpy(shrinked)
    #             imgs = self.prepare_images(imgs)
    #             result = imgs, np.array([1])
    #             if not self.example_generated:
    #                 for num, item in enumerate(imgs):
    #                     XISF.write(
    #                         os.path.join('C:\\git\\object_recognition\\examples', f"{num:03}.xisf"), item,
    #                         creator_app="My script v1.0",
    #                         codec='lz4hc', shuffle=True
    #                     )
    #                 self.example_generated = True
    #         else:
    #             shrinked = self.prepare_images(shrinked)
    #             result = shrinked, np.array([0])
    #         yield result
    #
    # def generate_batch(self, batch_size):
    #     series_generator = self.generate_series()
    #     while True:
    #         timestamps_batch = []
    #         batch = [next(series_generator) for _ in range(batch_size)]
    #         X_batch = np.array([item[0] for item in batch])
    #         y_batch = np.array([item[1] for item in batch])
    #         normalizer = (max(self.source_data.timestamps) - min(self.source_data.timestamps)).total_seconds()
    #         timestamps = [(max(self.source_data.timestamps) - ts).total_seconds() / normalizer for ts in self.source_data.timestamps]
    #         for _ in range(batch_size):
    #             timestamps_batch.append(timestamps)
    #         yield X_batch, y_batch

    @classmethod
    def get_max_image(cls, images):
        return np.amax(images, axis=0)


# class DataGenerator(tf.keras.utils.Sequence):
#     'Generates data for Keras'
#
#     def __init__(self, dataset, list_IDs=None, labels=None, batch_size=32, n_channels=1,
#                  n_classes=1, shuffle=False):
#         'Initialization'
#         self.dataset = dataset
#         self.dim = (len(self.dataset.raw_dataset), 54, 54, 1)
#         self.batch_size = batch_size
#         self.labels = labels
#         self.list_IDs = list_IDs
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.shuffle = shuffle
#         self.generator = self.dataset.generate_batch(self.batch_size)
#         self.on_epoch_end()
#
#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return len(self.list_IDs) // self.batch_size
#
#     def __getitem__(self, index):
#         'Generate one batch of data'
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
#
#         # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]
#
#         # Generate data
#         X, y = self.__data_generation(list_IDs_temp)
#
#         return X, y
#
#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#         self.indexes = np.arange(len(self.list_IDs))
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)
#
#     def __data_generation(self, _):
#         'Generates data containing batch_size samples'
#         X, y = next(self.generator)
#         y.shape = (self.batch_size, 1)
#         return X, y
#
#
# if __name__ == '__main__':
#     Dataset.crop_folder(
#         input_folder='C:\\Users\\bsolomin\\Astro\\Andromeda\\Pix_600\\registered\\Light_BIN-1_4544x3284_EXPOSURE-600.00s_FILTER-NoFilter_RGB',
#         output_folder='C:\\Users\\bsolomin\\Astro\\Andromeda\\Pix_600\\cropped\\'
#     )

import random

import time

import json
import matplotlib.pyplot as plt
import numpy as np
from auto_stretch.stretch import Stretch
import os
import pathlib
import datetime
import cv2
import keras


from xisf import XISF

class DatasetCreator:
    ZERO_TOLERANCE = 100
    SUB_SIZE = 254

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
            img_data = DatasetCreator.to_gray(img_data)
            y_boarders, x_boarders = DatasetCreator.crop_raw(img_data, to_do=False)
            y_boarders, x_boarders = DatasetCreator.crop_fine(
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
            img_data = DatasetCreator.to_gray(img_data)
            img_data = DatasetCreator.crop_image(img_data, y_boarders, x_boarders)
            img_data = DatasetCreator.stretch_image(img_data)
            img_data = np.array(img_data)
            img_data = np.float32(img_data)
            img_data.shape = *img_data.shape, 1
            XISF.write(
                os.path.join(output_folder, f"image_{num:04}.xisf"), img_data,
                creator_app="My script v1.0", image_metadata=img_meta[0], xisf_metadata=file_meta,
                codec='lz4hc', shuffle=True
            )

    @classmethod
    def shrink_folder(cls, input_folder, output_folder):
        if not os.path.exists(output_folder):
            pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
        file_list = [item for item in os.listdir(input_folder) if ".xisf" in item]
        fp = os.path.join(input_folder, file_list[0])
        xisf = XISF(fp)
        img_data = xisf.read_image(0)
        original_size = img_data.shape[:2]
        size = min(original_size)
        sizes = []
        while size > cls.SUB_SIZE:
            sizes.append(size)
            size //= 2
        sizes.append(cls.SUB_SIZE)
        sizes = np.array(sizes)
        size_offset_map = {}
        for size in sizes:
            y_offsets = np.arange(0, original_size[0], size//2)
            y_offsets = y_offsets[y_offsets + size <= original_size[0]]
            if y_offsets[-1] + size != original_size[0]:
                y_offsets = np.append(y_offsets, original_size[0] - size)
            x_offsets = np.arange(0, original_size[1], size//2)
            x_offsets = x_offsets[x_offsets + size <= original_size[1]]
            if x_offsets[-1] + size != original_size[1]:
                x_offsets = np.append(x_offsets, original_size[1] - size)
            size_offset_map.update({size: (y_offsets, x_offsets)})

        for size in sizes:
            print(f"Processing size {size}")
            for file_num, fp in enumerate(file_list, start=1):
                print(f"Processing image {fp}")
                xisf = XISF(os.path.join(input_folder, fp))
                file_meta = xisf.get_file_metadata()
                img_meta = xisf.get_images_metadata()
                img_data = xisf.read_image(0)
                timestamp = xisf.get_images_metadata()[0]["FITSKeywords"]["DATE-OBS"][0]['value']
                timestamp = datetime.datetime.strptime(timestamp.replace("T", " "), '%Y-%m-%d %H:%M:%S.%f')
                for y_num, y_offset in enumerate(size_offset_map[size][0]):
                    for x_num, x_offset in enumerate(size_offset_map[size][1]):
                        shrinked = DatasetCreator.crop_image(
                            img_data, (y_offset, y_offset + size), (x_offset, x_offset + size))
                        shrinked = cv2.resize(
                            shrinked, dsize=(cls.SUB_SIZE, cls.SUB_SIZE), interpolation=cv2.INTER_CUBIC)
                        y_ratio, x_ratio = cls.SUB_SIZE / shrinked.shape[0], cls.SUB_SIZE / shrinked.shape[1]
                        shrinked.shape = *shrinked.shape, 1
                        XISF.write(
                            os.path.join(output_folder, f"{fp}_{size}_{y_num}_{x_num}.xisf"), shrinked,
                            creator_app="My script v1.0", image_metadata=img_meta[0], xisf_metadata=file_meta,
                            codec='lz4hc', shuffle=True
                        )
                        with open(os.path.join(output_folder, f"{fp}_{size}_{y_num}_{x_num}.json"), "w") as fileo:
                            info = {
                                "image_num": file_num,
                                "timestamp": int(timestamp.timestamp()),
                                "size": int(size),
                                "x_offset": int(x_offset),
                                "y_offset": int(y_offset),
                                "x_ratio": x_ratio,
                                "y_ratio": y_ratio,
                            }
                            json.dump(info, fileo, indent=4)

    @classmethod
    def insert_star_by_coords(cls, image, star, coords):
        """

        :param image: filepath or 2 axis np.array where the star will be drawn.
        :param star: filepath or 2 axis np.array which represents star to be drawn.
        :param coords: tuple with (y, x) coordinates where the star will be drawn on the image.
        :return: 2 axis np.array.
        """
        if isinstance(star, str):
            star = XISF(star)
            star = star.read_image(0)[:, :, 0]
        elif isinstance(star, np.ndarray):
            assert len(star.shape) == 2
        else:
            raise ValueError(f"star parameter is expected to be a file path or 2 axis np.ndarray. got {star}")

        star_y_size, star_x_size = star.shape

        if isinstance(image, str):
            image = XISF(image)
            print(image.get_images_metadata())
            image = np.array(image.read_image(0)[:, :, 0])
        elif isinstance(image, np.ndarray):
            assert len(image.shape) == 2
        else:
            raise ValueError(f"image parameter is expected to be a file path or 2 axis np.ndarray. got {image}")
        image_y_size, image_x_size = image.shape
        x, y = coords
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

        y_slice = slice(y+cut_top-star_y_size//2, y-cut_bottom+star_y_size//2)
        x_slice = slice(x+cut_left-star_x_size//2, x-cut_right+star_x_size//2)

        image_to_correct = image[y_slice, x_slice]
        image[y_slice, x_slice] = np.maximum(
            star[cut_top:star_y_size - cut_bottom, cut_left:star_x_size - cut_right], image_to_correct)
        return image

    @classmethod
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
        if isinstance(star, str):
            star = XISF(star)
            star = star.read_image(0)[:, :, 0]
        elif isinstance(star, np.ndarray):
            assert len(star.shape) == 2
        else:
            raise ValueError(f"star parameter is expected to be a file path or 2 axis np.ndarray. got {star}")

        if isinstance(image, str):
            image = XISF(image)
            exposure_time = float(image.get_images_metadata()[0]["FITSKeywords"]["EXPTIME"][0]['value'])
            image = np.array(image.read_image(0)[:, :, 0])
        elif isinstance(image, np.ndarray):
            assert len(image.shape) == 2
            assert exposure_time
        else:
            raise ValueError(f"image parameter is expected to be a file path or 2 axis np.ndarray. got {image}")

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

    @classmethod
    def draw_object_on_image_series_numpy(cls, imgs, star_img):
        start_image_idx = random.randint(0, len(imgs))
        start_y = random.randint(0, cls.SUB_SIZE)
        start_x = random.randint(0, cls.SUB_SIZE)
        movement_vector = (random.randint(0, 10), random.randint(0, 10))
        y, x = start_y, start_x
        for img in imgs[start_image_idx:]:
            cls.calculate_star_form_on_single_image(img, star_img, (y, x), movement_vector, )






    @classmethod
    def draw_object_on_image_series(cls, input_folder, star, size,
                                    y_block, x_block, start_coords, movement_vector, output_folder=None):
        if isinstance(star, str):
            star = XISF(star)
            star = star.read_image(0)[:, :, 0]
        elif isinstance(star, np.ndarray):
            assert len(star.shape) == 2
        else:
            raise ValueError(f"star parameter is expected to be a file path or 2 axis np.ndarray. got {star}")
        file_list = [os.path.join(input_folder, item) for item in os.listdir(input_folder) if f"{size}_{y_block}_{x_block}.xisf" in item]
        start_ts = XISF(file_list[0]).get_images_metadata()[0]["FITSKeywords"]["DATE-OBS"][0]['value']
        start_ts = datetime.datetime.strptime(start_ts.replace("T", " "), '%Y-%m-%d %H:%M:%S.%f')
        # previous_ts = start_ts
        initial_image = file_list[0]
        initial_image = XISF(initial_image)
        initial_image = np.array(initial_image.read_image(0)[:, :, 0])
        print(initial_image.shape)
        for item in file_list:
            image = XISF(item)
            file_meta = image.get_file_metadata()
            img_meta = image.get_images_metadata()
            timestamp = img_meta[0]["FITSKeywords"]["DATE-OBS"][0]['value']
            timestamp = datetime.datetime.strptime(timestamp.replace("T", " "), '%Y-%m-%d %H:%M:%S.%f')
            exposure_time = float(img_meta[0]["FITSKeywords"]["EXPTIME"][0]['value'])
            image = np.array(image.read_image(0)[:, :, 0])
            inter_image_movement_vector = np.array(movement_vector) * (timestamp - start_ts).total_seconds() / 3600
            y, x = inter_image_movement_vector + np.array(start_coords)
            image = DatasetCreator.calculate_star_form_on_single_image(
                image, star, (round(y), round(x)), movement_vector, exposure_time=exposure_time
            )
            output_folder = output_folder if output_folder else input_folder
            if not os.path.exists(output_folder):
                pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
            # image = image - initial_image
            image.shape = *image.shape, 1
            XISF.write(
                os.path.join(output_folder, f"{os.path.basename(item)}"), image,
                creator_app="My script v1.0", image_metadata=img_meta[0], xisf_metadata=file_meta,
                codec='lz4hc', shuffle=True
            )

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32, 32, 32), n_channels=1,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.raw_dataset = np.array([])

    def load_raw_dataset(self, folder):
        file_list = [os.path.join(folder, item) for item in os.listdir(folder)]
        self.raw_dataset = np.array([XISF(item).read_image(0) for item in file_list])
        print(len(self.raw_dataset))
        print(self.raw_dataset.shape)
        print(self.raw_dataset.itemsize * self.raw_dataset.size)



    # def __len__(self):
    #     'Denotes the number of batches per epoch'
    #     return int(np.floor(len(self.list_IDs) / self.batch_size))
    #
    # def __getitem__(self, index):
    #     'Generate one batch of data'
    #     # Generate indexes of the batch
    #     indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
    #
    #     # Find list of IDs
    #     list_IDs_temp = [self.list_IDs[k] for k in indexes]
    #
    #     # Generate data
    #     X, y = self.__data_generation(list_IDs_temp)
    #
    #     return X, y

    # def on_epoch_end(self):
    #     'Updates indexes after each epoch'
    #     self.indexes = np.arange(len(self.list_IDs))
    #     if self.shuffle == True:
    #         np.random.shuffle(self.indexes)

    # def __data_generation(self, list_IDs_temp):
    #     'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
    #     # Initialization
    #     X = np.empty((self.batch_size, *self.dim, self.n_channels))
    #     y = np.empty((self.batch_size), dtype=int)
    #
    #     # Generate data
    #     for i, ID in enumerate(list_IDs_temp):
    #         # Store sample
    #         X[i,] = np.load('data/' + ID + '.npy')
    #
    #         # Store class
    #         y[i] = self.labels[ID]
    #
    #     return X, keras.utils.to_categorical(y, num_classes=self.n_classes)



if __name__ == '__main__':
    # DatasetCreator.crop_folder("C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\registered\\Light_BIN-1_4944x3284_EXPOSURE-300.00s_FILTER-NoFilter_RGB", "C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\cropped")
    # DatasetCreator.shrink_folder("C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\cropped", "C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\shrinked")

    # DatasetCreator.insert_star_by_coords(
    #     "C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\shrinked\\image_0001.xisf_224_1_19.xisf",
    #     "C:\\git\\object_recognition\\star_samples\\medium_star_1.xisf", (100, 100))

    # img = DatasetCreator.calculate_star_form_on_single_image(
    #     "C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\shrinked\\image_0001.xisf_224_1_19.xisf",
    #     "C:\\git\\object_recognition\\star_samples\\medium_star_1.xisf", (-5, 0), (200, 100)
    # )


    # DatasetCreator.draw_object_on_image_series(
    #     "C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\shrinked",
    #     "C:\\git\\object_recognition\\star_samples\\medium_star_1.xisf",
    #     224, 5, 5, (0, 0), (1, 1),
    #     "C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\experiment",
    # )
    DataGenerator(None, None).load_raw_dataset('C:\\Users\\bsolomin\\Astro\\Iris_2023\\Pix\\cropped')
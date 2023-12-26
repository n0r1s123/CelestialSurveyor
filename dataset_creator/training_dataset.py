import os
import random
import numpy as np

from PIL import Image

from .dataset import Dataset


class TrainingDataset(Dataset):
    def __init__(self, source_data, samples_folder: str):
        super(TrainingDataset, self).__init__(source_data)
        if not os.path.exists(samples_folder):
            raise ValueError(f"Samples folder '{samples_folder}' doesn't exist")
        self.object_samples = self.__load_samples(samples_folder)

    @classmethod
    def __load_samples(cls, samples_folder):
        file_list = [os.path.join(samples_folder, item) for item in os.listdir(samples_folder) if ".tif" in item]
        samples = np.array([np.array(Image.open(item)) for item in file_list])
        return samples

    def get_random_shrink(self, dataset_idx=0):
        size = 64
        exclusion_boxes = self.source_data[dataset_idx].exclusion_boxes
        generated = False

        # TODO: bad implementation need to rework
        while not generated:
            y = random.randint(0, self.source_data[dataset_idx].img_shape[0] - size)
            x = random.randint(0, self.source_data[dataset_idx].img_shape[1] - size)
            if exclusion_boxes is not None and len(exclusion_boxes) > 0:
                for box in exclusion_boxes:
                    x1, y1, x2, y2 = box
                    if (x1 - size <= x <= x2 + size) and (y1 - size <= y <= y2 + size):
                        break
                else:
                    generated = True
            else:
                generated = True
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
        image_y_size, image_x_size = image.shape[:2]
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
        # print(image[y_slice, x_slice].shape)
        # print(star[int(cut_top):int(star_y_size - cut_bottom), int(cut_left):int(star_x_size - cut_right)].shape)
        # print(image_to_correct.shape)
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

        dx = 0
        if x_move == y_move == 0:
            image = cls.insert_star_by_coords(image, star, (start_y, start_x))
            return image
        x_moves_per_y_moves = x_move / y_move

        for dy in range(round(y_move + 1)):
            if dy * x_moves_per_y_moves // 1 > dx:
                dx += 1
            elif dy * x_moves_per_y_moves // 1 < -dx:
                dx -= 1
            image = cls.insert_star_by_coords(image, star, (start_y + dy, start_x + dx))
        return image

    def draw_one_image_artefact(self, imgs):
        number_of_artefacts = random.choice(list(range(1, 5)) + [0] * 10)
        for _ in range(number_of_artefacts):
            y_shape, x_shape = imgs[0].shape[:2]
            star_img = random.choice(self.object_samples)
            start_image_idx = random.randint(0, len(imgs) - 1)
            y = random.randint(0, y_shape - 1)
            x = random.randint(0, x_shape - 1)
            object_factor = random.randrange(120, 300) / 300
            star_max = np.max(star_img)
            expected_max = np.average(imgs) + (np.max(imgs) - np.average(imgs)) * object_factor
            multiplier = expected_max / star_max
            star_img = star_img * multiplier

            # 0 means point like object which appears on the single image
            # 1 means satellite like object which appears on the single image
            is_satellite_like = random.randrange(0, 2)
            if not is_satellite_like:
                movement_vector = np.array([0, 0])
            else:
                movement_vector = np.array([random.randrange(1, 300) * 100, random.randrange(1, 300) * 100])

            imgs[start_image_idx] = self.calculate_star_form_on_single_image(
                imgs[start_image_idx], star_img, (y, x), movement_vector, 10000)
            imgs[start_image_idx] = self.calculate_star_form_on_single_image(
                imgs[start_image_idx], star_img, (y, x), - movement_vector, 10000)
        return imgs

    def draw_hot_pixels(self, imgs):
        probablity = 70
        result = []
        for img in imgs:
            if random.randrange(1, 101) < probablity:
                y_shape, x_shape = imgs[0].shape[:2]
                y = random.randint(0, y_shape - 1)
                x = random.randint(0, x_shape - 1)
                img[y, x] = 1
            result.append(img)
        result = np.array(result)
        return result

    def draw_variable_star(self, imgs, dataset_idx):
        old_images = np.copy(imgs)
        timestamps = [0.] + [(item - self.source_data[dataset_idx].timestamps[0]).total_seconds(
            ) for num, item in enumerate(self.source_data[dataset_idx].timestamps)]
        period = 1.5 * timestamps[-1]
        max_brightness = random.randrange(80, 101) / 100
        min_brightness = max_brightness - random.randrange(30, 61) / 100
        starting_phase = (random.randrange(0, 201) / 100) * np.pi
        y_shape, x_shape = imgs[0].shape[:2]
        y = random.randint(0, y_shape - 1)
        x = random.randint(0, x_shape - 1)
        # star_img = random.choice(self.object_samples)
        star_img = self.object_samples[-1]
        star_brightness = np.max(star_img)
        for num, (img, ts) in enumerate(zip(imgs, timestamps)):
            new_phaze = 2 * np.pi * ts / period + starting_phase
            new_brightness = np.sin(new_phaze) * (max_brightness - min_brightness) / 2 + (max_brightness + min_brightness) / 2
            brightness_multiplier = new_brightness / star_brightness
            new_star_image = star_img * brightness_multiplier
            new_img = self.calculate_star_form_on_single_image(
                img, new_star_image, (y, x), (0, 0), 10000)
            imgs[num] = new_img
        drawn = 1
        if (imgs == old_images).all():
            drawn = 0
        return imgs, drawn

    def draw_object_on_image_series_numpy(self, imgs, dataset_idx=0):
        min_total_movement = 5  # px
        old_images = np.copy(imgs)

        y_shape, x_shape = imgs[0].shape[:2]

        star_img = random.choice(self.object_samples)

        drawn = 0
        while not drawn:
            result = []
            start_image_idx = random.randint(0, len(imgs) - 1)
            start_y = random.randint(0, y_shape - 1)
            start_x = random.randint(0, x_shape - 1)
            object_factor = random.randrange(120, 301) / 300
            star_max = np.max(star_img)
            expected_max = np.average(imgs) + (np.max(imgs) - np.average(imgs)) * object_factor
            multiplier = expected_max / star_max
            star_img = star_img * multiplier

            # Calculate min and max movement vector length (pixels/hour)
            total_time = (self.source_data[dataset_idx].timestamps[-1] - self.source_data[dataset_idx].timestamps[0]
                          ).total_seconds()
            total_time /= 3600
            min_vector = max(min_total_movement / total_time, 0.5)
            max_vector = 50.  # pixels/hour
            vector_len = random.uniform(min_vector, max_vector)
            movement_angle = random.uniform(0., 2 * np.pi)
            movement_vector = np.array([np.sin(movement_angle), np.cos(movement_angle)]) * vector_len

            start_ts = self.source_data[dataset_idx].timestamps[start_image_idx]

            movement_vector = - movement_vector
            to_beginning_slice = slice(None, start_image_idx)
            for img, exposure, timestamp in zip(
                    imgs[to_beginning_slice][::-1], self.source_data[dataset_idx].exposures[to_beginning_slice][::-1],
                    self.source_data[dataset_idx].timestamps[to_beginning_slice][::-1]
            ):
                inter_image_movement_vector = np.array(movement_vector) * (timestamp - start_ts).total_seconds() / 3600
                y, x = inter_image_movement_vector + np.array([start_y, start_x])
                new_img = self.calculate_star_form_on_single_image(img, star_img, (y, x), movement_vector, exposure)
                result.append(new_img)

            result = result[::-1]

            to_end_slice = slice(start_image_idx, None, None)
            for img, exposure, timestamp in zip(
                    imgs[to_end_slice],
                    self.source_data[dataset_idx].exposures[to_end_slice],
                    self.source_data[dataset_idx].timestamps[to_end_slice]
            ):
                inter_image_movement_vector = np.array(movement_vector) * (timestamp - start_ts).total_seconds() / 3600
                y, x = inter_image_movement_vector + np.array([start_y, start_x])
                new_img = self.calculate_star_form_on_single_image(img, star_img, (y, x), movement_vector, exposure)
                result.append(new_img)
            result = np.array(result)

            drawn = 1
            if (result == old_images).all():
                drawn = 0
        if not drawn:
            print("not_drawn")
        return result, drawn

    def make_series(self, dataset_idx=0):
        imgs = self.get_shrinked_img_series(*self.get_random_shrink(dataset_idx), dataset_idx=dataset_idx)
        if random.randint(1, 101) > 70:
            what_to_draw = random.randrange(0, 100)
            if what_to_draw < 200:
                imgs, drawn = self.draw_object_on_image_series_numpy(imgs, dataset_idx=dataset_idx)
                res = drawn
            else:
                imgs, drawn = self.draw_variable_star(imgs, dataset_idx)
                res = drawn
        else:
            res = 0

        # if res == 0:
        if random.randint(0, 100) >= 80:
            imgs = self.draw_one_image_artefact(imgs)
        if random.randint(0, 100) >= 80:
            imgs = self.draw_hot_pixels(imgs)

        imgs = self.prepare_images(imgs)
        result = imgs, np.array([res])
        return result

    def make_batch(self, batch_size, save=False):
        dataset_idx = random.randrange(0, len(self.source_data))
        batch = [self.make_series(dataset_idx) for _ in range(batch_size)]
        X_batch = np.array([item[0] for item in batch])
        TS_batch = np.array([[
            self.source_data[dataset_idx].diff_timestamps,
            self.source_data[dataset_idx].normalized_timestamps] for _ in batch])
        TS_batch = np.swapaxes(TS_batch, 1, 2)
        y_batch = np.array([item[1] for item in batch])

        if save:
            for num, (bla_imgs, res) in enumerate(zip(X_batch, y_batch)):
                bla_imgs.shape = bla_imgs.shape[:3]
                bla_imgs = bla_imgs * 256
                new_frames = [Image.fromarray(frame).convert('L').convert('P') for frame in bla_imgs]
                new_frames[0].save(
                    f"{num}_{res[0]}.gif",
                    save_all=True,
                    append_images=new_frames[1:],
                    duration=200,
                    loop=0)
        # return [X_batch, TS_batch, X_batch[:, ::10], TS_batch[:, ::10]], y_batch
        return [X_batch, TS_batch], y_batch
        # return [X_batch, X_batch[:, ::4], TS_batch], y_batch
        # return X_batch, y_batch

    def batch_generator(self, batch_size):
        bla = True
        while True:
            yield self.make_batch(batch_size, bla)
            bla = False

import numpy as np
import os
import sys
import tifffile
import random
from PIL import Image
from decimal import Decimal
from typing import Sequence
from logger.logger import get_logger
from dataclasses import dataclass
import datetime
from bs4 import BeautifulSoup
from typing import Optional
from backend.consuming_functions import measure_execution_time

from backend.source_data_v2 import SourceDataV2, CHUNK_SIZE


logger = get_logger()


MIN_TOTAL_MOVEMENT = 5  # px


@dataclass
class RandomObject:
    start_y: int
    start_x: int
    start_frame_idx: int
    movement_vector: np.ndarray
    brightness_above_noize: float
    star_sample: np.ndarray
    exposure: Decimal
    timestamps: Sequence[datetime.datetime]


class TrainingSourceDataV2(SourceDataV2):
    SAMPLES_FOLDER = os.path.join(sys.path[1], "star_samples")

    def __init__(self, file_list: list[str], to_debayer: bool = False, number_of_images: Optional[int] = None) -> None:
        file_list = file_list[:number_of_images]
        super().__init__(file_list, to_debayer)
        self.exclusion_boxes = []
        logger.log.info("Loading star samples")
        self.star_samples = self._load_star_samples()

    @classmethod
    def _load_star_samples(cls) -> np.ndarray:
        file_list = [
            os.path.join(cls.SAMPLES_FOLDER, item) for item in os.listdir(cls.SAMPLES_FOLDER) if ".tif" in item]
        star_samples = np.array([np.array(tifffile.tifffile.imread(item)) for item in file_list])
        if len(star_samples.shape) == 3:
            star_samples = np.reshape(star_samples, (*star_samples.shape, 1))
        return star_samples

    def __get_exclusion_boxes_paths(self):
        folders = {os.path.dirname(header.file_name) for header in self.headers}
        exclusion_boxes_files = []
        for folder in folders:
            if "annotations.xml" in os.listdir(folder):
                exclusion_boxes_files.append(os.path.join(folder, "annotations.xml"))
        return exclusion_boxes_files

    def load_exclusion_boxes(self):
        # Reading data from the xml file
        img_shape = self.shape[1:]
        all_boxes = []

        for fp in self.__get_exclusion_boxes_paths():
            with open(fp, 'r') as f:
                data = f.read()
            bs_data = BeautifulSoup(data, 'xml')
            boxes = []
            width = float(bs_data.find('image').get("width"))
            height = float(bs_data.find('image').get("height"))
            if img_shape is None:
                x_mult, y_mult = 1, 1
            else:
                y_shape, x_shape, *_ = img_shape
                y_mult = y_shape / height
                x_mult = x_shape / width

            for tag in bs_data.find_all('box', {'label': 'Asteroid'}):
                xtl = round(float(tag.get("xtl")) * x_mult) - 100
                ytl = round(float(tag.get("ytl")) * y_mult) - 100
                xbr = round(float(tag.get("xbr")) * x_mult) + 100
                ybr = round(float(tag.get("ybr")) * y_mult) + 100
                boxes.append((xtl, ytl, xbr, ybr))

            all_boxes.extend(boxes)
        self.exclusion_boxes = np.array(all_boxes)

    def get_random_shrink(self) -> np.ndarray:
        generated = False
        while not generated:
            y = random.randint(0, self.shape[1] - CHUNK_SIZE)
            x = random.randint(0, self.shape[2] - CHUNK_SIZE)
            for box in self.exclusion_boxes:
                x1, y1, x2, y2 = box
                if (x1 - CHUNK_SIZE <= x <= x2 + CHUNK_SIZE) and (y1 - CHUNK_SIZE <= y <= y2 + CHUNK_SIZE):
                    break
            else:
                res = np.copy(self.images[:, y:y + CHUNK_SIZE, x:x + CHUNK_SIZE])
                return res

    @classmethod
    def insert_star_by_coords(cls, image, star, coords):
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
            star[int(cut_top):int(star_y_size - cut_bottom), int(cut_left):int(star_x_size - cut_right)],
            image_to_correct)
        return image

    @classmethod
    def calculate_star_form_on_single_image(cls, image, star, start_coords, movement_vector, exposure_time=None):
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

    def generate_timestamps(self) -> Sequence[datetime.datetime]:
        # generate random timestamps emulating timestamps of observations sessions staring from now
        # each session should have at least 5 timestamps. If there are less than 5 timestamps - only one session will be
        # generated. number of sessions is random. Interval between timestamps is equal to the same random exposure time
        # applicable for all intervals plus some random offset
        ts_num = len(self.images)
        max_sessions_num = min((ts_num - 1) // 5 + 1, 4)
        sessions_num = random.randrange(0, max_sessions_num)
        exposures = (0.25, 0.5, *tuple(range(1, 11)))
        exposure = random.choice(exposures) * 60
        new_session_starts = []
        bla_timestamps = list(range(ts_num))
        for _ in range(sessions_num):
            temp_session_start = random.choice(bla_timestamps)
            bla_timestamps.pop(bla_timestamps.index(temp_session_start))
            new_session_starts.append(temp_session_start)
        new_session_starts.sort()
        start_ts = datetime.datetime.now()
        timestamps = []
        max_inter_exposure = 20 * 60  # 20 minutes
        added_days = 0
        for num in range(ts_num):
            if num == 0:
                next_timestamp = datetime.datetime.now()
            elif num in new_session_starts:
                added_days += 1
                next_timestamp = start_ts + datetime.timedelta(
                    days=added_days, hours=random.randrange(0, 3), seconds=random.randrange(0, 3600))
            else:
                next_timestamp = timestamps[-1] + datetime.timedelta(
                    seconds=exposure + random.randrange(1, max_inter_exposure + 1))
            timestamps.append(next_timestamp)
        return timestamps, exposure

    def generate_random_objects(self):
        start_y = random.randrange(0, CHUNK_SIZE)
        start_x = random.randrange(0, CHUNK_SIZE)
        start_frame_idx = random.randrange(0, len(self.images))
        timestamps, exposure = self.generate_timestamps()
        brightness_above_noize = float(random.randrange(400, 1001)) / 1000
        star_sample = random.choice(self.star_samples)
        total_time = (timestamps[-1] - timestamps[0]).total_seconds()
        total_time /= 3600
        min_vector = max(MIN_TOTAL_MOVEMENT / total_time, 0.5)
        max_vector = 50.  # pixels/hour
        vector_len = random.uniform(min_vector, max_vector)
        movement_angle = random.uniform(0., 2 * np.pi)
        movement_vector = np.array([np.sin(movement_angle), np.cos(movement_angle)]) * vector_len
        return RandomObject(start_y, start_x, start_frame_idx, movement_vector, brightness_above_noize, star_sample,
                           exposure, timestamps)

    def draw_object_on_image_series_numpy(self, rand_obg):
        imgs = self.get_random_shrink()
        old_images = np.copy(imgs.copy)
        drawn = 0

        while not drawn:
            result = []
            noise_level = self.estimate_image_noize_level(imgs)
            signal_space = 1 - noise_level
            expected_star_max = signal_space * rand_obg.brightness_above_noize + noise_level
            star_max = np.amax(rand_obg.star_sample)
            multiplier = expected_star_max / star_max
            star_img = rand_obg.star_sample * multiplier

            movement_vector = rand_obg.movement_vector
            to_beginning_slice = slice(None, rand_obg.start_frame_idx)
            start_ts = rand_obg.timestamps[rand_obg.start_frame_idx]
            for img, timestamp in zip(
                    imgs[to_beginning_slice][::-1],
                    rand_obg.timestamps[to_beginning_slice][::-1]
            ):
                inter_image_movement_vector = np.array(movement_vector) * (timestamp - start_ts).total_seconds() / 3600
                y, x = inter_image_movement_vector + np.array([rand_obg.start_y, rand_obg.start_x])

                new_img = self.calculate_star_form_on_single_image(img, star_img, (y, x), movement_vector, rand_obg.exposure)
                result.append(new_img)

            result = result[::-1]

            to_end_slice = slice(rand_obg.start_frame_idx, None, None)
            for img, timestamp in zip(
                    imgs[to_end_slice],
                    rand_obg.timestamps[to_end_slice]
            ):
                inter_image_movement_vector = np.array(movement_vector) * (timestamp - start_ts).total_seconds() / 3600
                y, x = inter_image_movement_vector + np.array([rand_obg.start_y, rand_obg.start_x])
                new_img = self.calculate_star_form_on_single_image(img, star_img, (y, x), movement_vector, rand_obg.exposure)
                result.append(new_img)
            result = np.array(result)
            #
            # import cv2
            # for img1, img2 in zip(old_images, result):
            #     cv2.imshow('img', img1)
            #     cv2.waitKey(0)
            #     cv2.imshow('img', img2)
            #     cv2.waitKey(0)

            drawn = 1
            if (result == old_images).all():
                drawn = 0
                rand_obg = self.generate_random_objects()
                print("Object not drawn")
        return result, drawn

    def draw_variable_star(self, rand_obj):
        imgs = np.copy(self.get_random_shrink())
        old_images = np.copy(imgs)
        # TODO: Refactor
        original_timestamps = rand_obj.timestamps
        timestamps = [0.] + [(item - original_timestamps[0]).total_seconds(
        ) for num, item in enumerate(original_timestamps)]
        period = 1.5 * timestamps[-1]
        max_brightness = random.randrange(80, 101) / 100
        min_brightness = max_brightness - random.randrange(30, 61) / 100
        starting_phase = (random.randrange(0, 201) / 100) * np.pi
        y_shape, x_shape = imgs[0].shape[:2]
        y = random.randint(0, y_shape - 1)
        x = random.randint(0, x_shape - 1)
        # star_img = random.choice(self.object_samples)
        star_img = self.star_samples[-1]
        star_brightness = np.max(star_img)
        for num, (img, ts) in enumerate(zip(imgs, timestamps)):
            new_phaze = 2 * np.pi * ts / period + starting_phase
            new_brightness = np.sin(new_phaze) * (max_brightness - min_brightness) / 2 + (
                        max_brightness + min_brightness) / 2
            brightness_multiplier = new_brightness / star_brightness
            new_star_image = star_img * brightness_multiplier
            new_img = self.calculate_star_form_on_single_image(img, new_star_image, (y, x), (0, 0), 10000)
            imgs[num] = new_img
        drawn = 1
        if (imgs == old_images).all():
            drawn = 0
        return imgs, drawn

    def draw_one_image_artefact(self, imgs):
        number_of_artefacts = random.choice(list(range(1, 5)) + [0] * 10)
        for _ in range(number_of_artefacts):
            y_shape, x_shape = imgs[0].shape[:2]
            star_img = random.choice(self.star_samples)
            start_image_idx = random.randint(0, len(imgs) - 1)
            y = random.randint(0, y_shape - 1)
            x = random.randint(0, x_shape - 1)
            object_factor = random.randrange(120, 300) / 300
            star_max = np.amax(star_img)
            expected_max = np.average(imgs) + (np.max(imgs) - np.average(imgs)) * object_factor
            if star_max == 0:
                multiplier = 1
            else:
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

    @staticmethod
    def draw_hot_pixels(imgs, dead=False):
        imgs = np.copy(imgs)
        probablity = 20
        result = []
        for img in imgs:
            if random.randrange(1, 101) < probablity:
                y_shape, x_shape = imgs[0].shape[:2]
                y = random.randint(0, y_shape - 1)
                x = random.randint(0, x_shape - 1)
                img[y, x] = 0 if dead else random.randrange(50, 101) / 100.
            result.append(img)
        result = np.array(result)
        return result


class TrainingDatasetV2:
    def __init__(self, source_datas: Sequence[TrainingSourceDataV2]):
        self.source_datas = source_datas

    def make_series(self, source_data):
        rand_obg = source_data.generate_random_objects()
        if random.randint(1, 101) > 50:
            what_to_draw = random.randrange(0, 100)
            if what_to_draw < 200:
                imgs, res = source_data.draw_object_on_image_series_numpy(rand_obg)
            else:
                imgs, drawn = source_data.draw_variable_star(rand_obg)
                res = drawn
        else:
            res = 0
            imgs = source_data.get_random_shrink()

        if random.randint(0, 100) >= 80:
            imgs = source_data.draw_one_image_artefact(imgs)
        if random.randint(0, 100) >= 70:
            imgs = source_data.draw_hot_pixels(imgs, bool(random.randrange(0, 2)))

        imgs = source_data.prepare_images(imgs)
        imgs, timestamps = source_data.adjust_chunks_to_min_len(imgs, rand_obg.timestamps, min_len=5)

        # TODO: Timestamp normalization if required
        # slow input
        # average frame of each 4 frames
        slow_images = []
        # for i in range(len(imgs)):
        #     if i % 4 == 0:
        #         slow_images.append(np.average(imgs[i:i + 4], axis=0))
        # slow_images = np.array(slow_images)
        # return imgs, slow_images, np.array([res])
        return imgs, None, np.array([res])

    def make_batch(self, batch_size, save=False):
        source_data = random.choice(self.source_datas)
        batch = [self.make_series(source_data) for _ in range(batch_size)]
        X_fast_batch = np.array([item[0] for item in batch])
        # X_slow_batch = np.array([item[1] for item in batch])
        y_batch = np.array([item[2] for item in batch])

        if save:
            for num, (bla_imgs, res) in enumerate(zip(X_fast_batch, y_batch)):
                bla_imgs.shape = bla_imgs.shape[:3]
                bla_imgs = bla_imgs * 256
                new_frames = [Image.fromarray(frame).convert('L').convert('P') for frame in bla_imgs]
                new_frames[0].save(
                    f"{num}_{res[0]}.gif",
                    save_all=True,
                    append_images=new_frames[1:],
                    duration=200,
                    loop=0)
        # return [X_fast_batch, X_slow_batch], y_batch
        return X_fast_batch, y_batch

    def batch_generator(self, batch_size):
        bla = True
        while True:
            yield self.make_batch(batch_size, bla)
            bla = False


if __name__ == '__main__':
    pass

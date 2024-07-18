import cv2
import datetime
import json
import numba
import numpy as np
import os
import random
import sys
import tifffile


from dataclasses import dataclass
from decimal import Decimal
from PIL import Image
from typing import Optional, Sequence, Generator

from backend.source_data_v2 import SourceDataV2, CHUNK_SIZE
from logger.logger import get_logger


logger = get_logger()

MIN_TOTAL_MOVEMENT = 5  # px


@dataclass
class RandomObject:
    """
    Represents a random object to be drawn on the image series.

    Attributes:
        start_y (int): The starting y-coordinate of the object.
        start_x (int): The starting x-coordinate of the object.
        start_frame_idx (int): The starting frame index of the object.
        movement_vector (np.ndarray): The movement vector of the object.
        brightness_above_noize (float): The brightness level above noise.
        star_sample (np.ndarray): The sample of the star.
        exposure (Decimal): The exposure value of the object.
        timestamps (Sequence[datetime.datetime]): The timestamps associated with the object.
    """
    start_y: int
    start_x: int
    start_frame_idx: int
    movement_vector: np.ndarray
    brightness_above_noize: float
    star_sample: np.ndarray
    exposure: Decimal
    timestamps: list[datetime.datetime]


class TrainingSourceDataV2(SourceDataV2):
    """
    Class to manage image data. One instance of this class manages one object captured.
    There are some methods to generate synthetic data for AI model training.
    """

    SAMPLES_FOLDER = os.path.join(sys.path[1], "star_samples")
    OBJ_TYPE_STAR = "star"
    OBJ_TYPE_ALL = "all"
    OBJ_TYPE_COMET = "comet"

    def __init__(self, to_debayer: bool = False, number_of_images: Optional[int] = None) -> None:
        super().__init__(to_debayer)
        self.number_of_images = number_of_images
        self.exclusion_boxes = []
        logger.log.info("Loading star samples")
        self.star_samples, self.comet_samples = self._load_star_samples()

    @classmethod
    def _load_star_samples(cls) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Loads star and comet samples from the SAMPLES_FOLDER directory.

        Returns:
        Tuple containing a list of star samples and a list of comet samples, both as numpy arrays.
        """
        file_list_stars = [
            os.path.join(cls.SAMPLES_FOLDER, item) for item in os.listdir(cls.SAMPLES_FOLDER)
            if ".tif" in item.lower() and "star" in item.lower()]
        file_list_comets = [
            os.path.join(cls.SAMPLES_FOLDER, item) for item in os.listdir(cls.SAMPLES_FOLDER)
            if ".tif" in item.lower() and "comet" in item.lower()]
        star_samples = [np.array(tifffile.tifffile.imread(item)) for item in file_list_stars]
        comet_samples = [np.array(tifffile.tifffile.imread(item)) for item in file_list_comets]
        for num, sample in enumerate(star_samples):
            if len(sample.shape) == 3:
                star_samples[num] = np.reshape(sample, sample.shape[:2])
            if sample.shape[0] % 2 == 1:
                star_samples[num] = np.delete(star_samples[num], 0, axis=0)

            if sample.shape[1] % 2 == 1:
                star_samples[num] = np.delete(star_samples[num], 0, axis=1)
            if sample.dtype == 'uint16':
                star_samples[num] = star_samples[num].astype(np.float32) / 65535

        for num, sample in enumerate(comet_samples):
            if len(sample.shape) == 3:
                comet_samples[num] = np.reshape(sample, sample.shape[:2])
            if sample.shape[0] % 2 == 1:
                comet_samples[num] = np.delete(comet_samples[num], 0, axis=0)

            if sample.shape[1] % 2 == 1:
                comet_samples[num] = np.delete(comet_samples[num], 0, axis=1)
            if sample.dtype == 'uint16':
                comet_samples[num] = comet_samples[num].astype(np.float32) / 65535

        return star_samples, comet_samples

    def __get_exclusion_boxes_paths(self) -> list[str]:
        """
        Get the paths of exclusion boxes files based on the directories of the headers.

        Returns:
        List of paths to exclusion boxes files.
        """
        folders = {os.path.dirname(header.file_name) for header in self.headers}
        exclusion_boxes_files = []
        for folder in folders:
            if "exclusion_boxes.json" in os.listdir(folder):
                exclusion_boxes_files.append(os.path.join(folder, "exclusion_boxes.json"))
        return exclusion_boxes_files

    def load_exclusion_boxes(self, force_rebuild: bool = False, magnitude_limit: float = 18.0) -> None:
        """
        Loads exclusion. Exclusion boxes contain real asteroids in the training data.
        These boxes shouldn't be used for training due to we cannot say if there was synthetic object drawn.

        Parameters:
        - force_rebuild: bool, optional, default is False
            If True, forces a rebuild of the exclusion boxes.
        - magnitude_limit: float, optional, default is 18.0
            The magnitude limit for exclusion boxes.

        Returns:
        None
        """
        all_boxes = []
        file_paths = self.__get_exclusion_boxes_paths()
        if len(file_paths) > 0 and not force_rebuild:
            logger.log.info("Loading exclusion boxes...")
            for fp in file_paths:
                with open(fp, 'r') as fileo:
                    exclusion_boxes = json.load(fileo)
                    all_boxes.extend(exclusion_boxes)
            all_boxes = np.array(all_boxes)
            self.exclusion_boxes = all_boxes
        else:
            logger.log.info("Making exclusion boxes...")
            self.make_exclusion_boxes(magnitude_limit=magnitude_limit)

    def make_exclusion_boxes(self, magnitude_limit: float = 18.0) -> None:
        """
        Create exclusion boxes. It requests the list of known objects for the first image of each imaging session.
        Creates JSON file with the exclusion boxes not to request them again.

        Parameters:
        - magnitude_limit: float, optional, default is 18.0
            The magnitude limit for exclusion boxes.

        Returns:
        None
        """
        start_session_idx = 0
        session_timestamps = []
        exclusion_boxes = []
        for num, header in enumerate(self.headers):
            if header.timestamp - self.headers[start_session_idx].timestamp > datetime.timedelta(hours=14):
                session_timestamps.append((start_session_idx, num - 1))
                start_session_idx = num
        else:
            session_timestamps.append((start_session_idx, len(self.headers) - 1))

        for start, end in session_timestamps:
            start_asteroids, start_comets = self.fetch_known_asteroids_for_image(start, magnitude_limit=magnitude_limit)
            end_asteroids, end_comets = self.fetch_known_asteroids_for_image(end, magnitude_limit=magnitude_limit)
            start_objects = start_asteroids + start_comets
            end_objects = end_asteroids + end_comets
            for start_object in start_objects:
                for end_object in end_objects:
                    if start_object.name == end_object.name:
                        start_x, start_y = start_object.pixel_coordinates
                        end_x, end_y = end_object.pixel_coordinates
                        start_x = int(start_x)
                        start_y = int(start_y)
                        end_x = int(end_x)
                        end_y = int(end_y)
                        exclusion_boxes.append((
                            max(min(start_x, end_x) - 50, 0),
                            max(min(start_y, end_y) - 50, 0),
                            min(max(start_x, end_x) + 50, self.shape[2] - 1),
                            min(max(start_y, end_y) + 50, self.shape[1] - 1),
                        ))
        folder = os.path.dirname(self.headers[0].file_name)
        file_name = os.path.join(folder, "exclusion_boxes.json")
        with open(file_name, 'w') as fileo:
            json.dump(exclusion_boxes, fileo)
        exclusion_boxes = np.array(exclusion_boxes)
        self.exclusion_boxes = exclusion_boxes

    def show_exclusion_boxes(self) -> None:
        """
        Display exclusion boxes on an image. (Debug method)
        """
        max_image = np.copy(self.max_image)
        max_image = cv2.cvtColor(max_image, cv2.COLOR_GRAY2BGR)
        for xlt, ylt, xbr, ybr in self.exclusion_boxes:
            max_image = cv2.rectangle(max_image, (xlt, ylt), (xbr, ybr), (255, 0, 0), 3)
        small = cv2.resize(max_image, (0, 0), fx=0.4, fy=0.4)
        cv2.imshow("Exclusion boxes", small)
        cv2.waitKey(0)

    def get_random_shrink(self) -> np.ndarray:
        """
        Generate a random image chunk that does not overlap with exclusion boxes.

        Returns:
        Numpy array representing the random image chunk.
        """
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
                res = np.reshape(res, res.shape[:3])
                return res

    @staticmethod
    def insert_star_by_coords(image: np.ndarray, star: np.ndarray, coords: tuple[int, int]) -> np.ndarray:
        """
        Inserts a sample of the star onto another image at specified coordinates.

        Args:
        image (np.ndarray): The image to insert the star image into.
        star (np.ndarray): The star image to be inserted.
        coords (tuple[int, int]): The coordinates to place the star image.

        Returns:
        np.ndarray: The image with the star image inserted.
        """
        return insert_star_by_coords(image, star, coords)

    @classmethod
    def calculate_star_form_on_single_image(
            cls, image: np.ndarray, star: np.ndarray, start_coords: tuple[int, int],
            movement_vector: np.ndarray, exposure_time: Optional[Decimal] = None) -> np.ndarray:
        """
        Calculate the form of a star sample on a single image based on the star sample, coordinates, movement vector,
        and exposure time. The idea is to emulate object sample movement during the single exposure.

        Args:
            image (np.ndarray): The image to insert the star form into.
            star (np.ndarray): The star sample image to be inserted.
            start_coords (Tuple[int, int]): The starting coordinates to place the star image.
            movement_vector (np.ndarray): The movement vector of the star image.
            exposure_time (Optional[float], optional): The exposure time of the image. Defaults to None.

        Returns:
            np.ndarray: The image with the star form inserted.
        """
        return calculate_star_form_on_single_image(image, star, start_coords, movement_vector, exposure_time)

    def generate_timestamps(self) -> tuple[list[datetime.datetime], Decimal]:
        """
        Generate random timestamps emulating timestamps of observation sessions.

        Returns:
            Tuple: A tuple containing a list of generated timestamps and a random exposure time.
        """
        timestamps, exposure = generate_timestamps(len(self.images))
        exposure = Decimal(exposure)
        timestamps = [datetime.datetime.utcfromtimestamp(ts) for ts in timestamps]
        return timestamps, exposure

    def generate_random_objects(self, obj_type: str = "star") -> RandomObject:
        """
        Generates random objects based on the provided object type.

        Parameters:
            obj_type (str): The type of object to generate ("star" or "comet"). Default is "star".

        Returns:
            RandomObject: An object containing the generated random properties.
        """
        if obj_type == self.OBJ_TYPE_STAR:
            samples = self.star_samples
        elif obj_type == self.OBJ_TYPE_COMET:
            samples = self.comet_samples
        elif obj_type == self.OBJ_TYPE_ALL:
            samples = self.comet_samples + self.star_samples
        else:
            raise ValueError(f"Unknown object type: {obj_type}")

        (start_y, start_x, start_frame_idx, movement_vector, brightness_above_noize, star_sample, exposure, timestamps
         ) = generate_random_objects(len(self.images), samples)
        timestamps = [datetime.datetime.utcfromtimestamp(ts) for ts in timestamps]
        return RandomObject(start_y, start_x, start_frame_idx, movement_vector,
                            brightness_above_noize, star_sample, Decimal(exposure), timestamps)

    def draw_object_on_image_series_numpy(self, rand_obg: RandomObject) -> tuple[np.ndarray, int]:
        """
        Draws a random object on a random image series. The idea is to emulate object sample movement on the image
            series.
        We choose random object sample, random movement vector (in pixels per hour), and random exposure time.
        Then we need to draw the object on the image series taking in account timestamps and exposure time.

        Args:
            rand_obg (RandomObject): The random object to draw on the image series.

        Returns:
            np.ndarray: The image series with the random object drawn on it.
        """
        imgs = self.get_random_shrink()
        imgs = np.reshape(np.copy(imgs), (imgs.shape[:3]))
        old_images = np.copy(imgs)
        drawn = 0
        while not drawn:
            noise_level = self.estimate_image_noize_level(imgs)
            signal_space = 1 - noise_level
            expected_star_max = signal_space * rand_obg.brightness_above_noize + noise_level
            star_max = np.amax(rand_obg.star_sample)
            multiplier = expected_star_max / star_max
            star_img = rand_obg.star_sample * multiplier

            movement_vector = rand_obg.movement_vector

            # Here we solve the problem object coordinates and movement vector selection to guarantee that the object
            # will appear in the series. Instead of choosing coordinates on the first image and calculating possible
            # movement vector - we choose coordinates on the image in the middle of the series, choose random movement
            # vector and insert it on each emage from the middle to the beginning and from the middle to the end.

            # Draw the object on the image series moving backwards from the start frame to the beginning of the series.
            to_beginning_slice = slice(None, rand_obg.start_frame_idx)
            start_ts = rand_obg.timestamps[rand_obg.start_frame_idx]
            for img, timestamp in zip(
                    imgs[to_beginning_slice][::-1],
                    rand_obg.timestamps[to_beginning_slice][::-1]
            ):
                inter_image_movement_vector = np.array(movement_vector) * (timestamp - start_ts).total_seconds() / 3600
                y, x = inter_image_movement_vector + np.array([rand_obg.start_y, rand_obg.start_x])
                if y + star_img.shape[0] < 0 or y - star_img.shape[0] > img.shape[0]:
                    break
                if x + star_img.shape[1] < 0 or x - star_img.shape[1] > img.shape[1]:
                    break

                new_img = self.calculate_star_form_on_single_image(
                    img, star_img, (y, x), movement_vector, rand_obg.exposure)
                img[:] = new_img

            # Draw the object on the image series moving from the start frame to the end of the series.
            to_end_slice = slice(rand_obg.start_frame_idx, None, None)
            for img, timestamp in zip(
                    imgs[to_end_slice],
                    rand_obg.timestamps[to_end_slice]
            ):
                inter_image_movement_vector = np.array(movement_vector) * (timestamp - start_ts).total_seconds() / 3600
                y, x = inter_image_movement_vector + np.array([rand_obg.start_y, rand_obg.start_x])
                if y + star_img.shape[0] < 0 or y - star_img.shape[0] > img.shape[0]:
                    break
                if x + star_img.shape[1] < 0 or x - star_img.shape[1] > img.shape[1]:
                    break
                new_img = self.calculate_star_form_on_single_image(
                    img, star_img, (y, x), movement_vector, rand_obg.exposure)
                img[:] = new_img
            result = imgs

            drawn = 1
            if (result == old_images).all():
                drawn = 0
                rand_obg = self.generate_random_objects(self.OBJ_TYPE_ALL)
        return result, drawn

    def draw_variable_star(self, rand_obj: RandomObject) -> tuple[np.ndarray, int]:
        """
        Emulates variable stars on image series.

        Note: For future debugging. Is not used in the current implementation.
        """
        raise NotImplementedError("This function needs to be reviewed and refactored.")
        # TODO: Refactor
        imgs = np.copy(self.get_random_shrink())
        old_images = np.copy(imgs)

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

    # @measure_execution_time
    def draw_one_image_artefact(self, imgs: np.ndarray) -> np.ndarray:
        """
        Draws one image artefact. Something like cosmic rays or satellite/airplane tracks.

        Args:
            imgs (np.ndarray): The input images.

        Returns:
            np.ndarray: The modified images.
        """
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
    def draw_hot_pixels(imgs: np.ndarray, dead: bool = False) -> np.ndarray:
        """
        Draws hot pixels on the images.

        Args:
            imgs: The images to draw hot pixels on.
            dead (bool): If True, sets the pixel value to 0, otherwise adjusts brightness.

        Returns:
            np.ndarray: The images with hot pixels drawn on them.
        """
        imgs = np.copy(imgs)
        probablity = random.randrange(10, 51)
        brightness = random.randrange(90, 101) / 100.
        result = []
        for img in imgs:
            if random.randrange(1, 101) < probablity:
                y_shape, x_shape = imgs[0].shape[:2]
                y = random.randint(0, y_shape - 1)
                x = random.randint(0, x_shape - 1)
                img[y, x] = 0 if dead else brightness
            result.append(img)
        result = np.array(result)
        return result

    def draw_hot_stars(self, imgs: np.ndarray) -> np.ndarray:
        """
        Draws hot stars on the images. Like it's done in draw_hot_pixels function, but for object samples,
        not for single pixels.

        Args:
            imgs: The images to draw hot stars on.

        Returns:
            np.ndarray: The images with hot stars drawn on them.
        """
        imgs = np.copy(imgs)
        probability = random.randrange(10, 51)
        brightness = random.randrange(80, 101) / 100.
        star_img = random.choice(self.star_samples)
        star_img *= brightness / np.amax(star_img)
        result = []
        for img in imgs:
            if random.randrange(1, 101) < probability:
                y_shape, x_shape = imgs[0].shape[:2]
                y = random.randint(0, y_shape - 1)
                x = random.randint(0, x_shape - 1)
                img = self.insert_star_by_coords(img, star_img, (y, x))
            result.append(img)
        result = np.array(result)
        return result


class TrainingDatasetV2:
    """
    Represents all the data used for training the model.
    """
    def __init__(self, source_datas: Sequence[TrainingSourceDataV2]):
        self.source_datas = source_datas

    @staticmethod
    def make_series(source_data: TrainingSourceDataV2) -> tuple[np.ndarray, int]:
        """
        Creates a series of images with random objects for training purposes.

        Args:
            source_data (TrainingSourceDataV2): The data source for generating random objects.

        Returns:
            Tuple containing the generated image series and a result value.
        """
        rand_obg = source_data.generate_random_objects(obj_type=source_data.OBJ_TYPE_ALL)
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

        if random.randint(0, 100) >= 90:
            imgs = source_data.draw_one_image_artefact(imgs)
        if random.randint(0, 100) >= 90:
            if random.randint(0, 100) >= 50:
                imgs = source_data.draw_hot_stars(imgs)
            else:
                imgs = source_data.draw_hot_pixels(imgs, bool(random.randrange(0, 2)))
        imgs = source_data.prepare_images(imgs)
        imgs, timestamps = source_data.adjust_chunks_to_min_len(imgs, rand_obg.timestamps, min_len=5)
        return imgs, res

    def make_batch(self, batch_size: int, save: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Creates a batch of images with random objects for training purposes.

        Args:
            batch_size (int): The size of the batch.
            save (bool): If True, saves the generated images as GIFs.

        Returns:
            Tuple containing the generated image series and a result value.
        """
        source_data = random.choice(self.source_datas)
        batch = [self.make_series(source_data) for _ in range(batch_size)]
        x_fast_batch = np.array([item[0] for item in batch])
        y_batch = np.array([item[1] for item in batch])

        # for debug purposes - it's possible to review some samples of synthetic data used for training
        if save:
            for num, (bla_imgs, res) in enumerate(zip(x_fast_batch, y_batch)):
                bla_imgs.shape = bla_imgs.shape[:3]
                bla_imgs = bla_imgs * 256
                new_frames = [Image.fromarray(frame).convert('L').convert('P') for frame in bla_imgs]
                new_frames[0].save(
                    f"{num}_{res[0]}.gif",
                    save_all=True,
                    append_images=new_frames[1:],
                    duration=200,
                    loop=0)
        return x_fast_batch, y_batch

    def batch_generator(self, batch_size: int) -> Generator[tuple[np.ndarray, np.ndarray]]:
        """
        Generator function that yields batches of image data and corresponding labels.

        Args:
            batch_size (int): The size of each batch.

        Yields:
            tuple[np.ndarray, np.ndarray]: A tuple containing the batch of images and their corresponding labels.
        """
        bla = False
        i = 0
        while True:
            # for debug purposes - it's possible to review some samples of synthetic data used for training
            if i == 450:
                bla = True
            yield self.make_batch(batch_size, bla)
            i += 1
            bla = False


@numba.jit(nopython=True, fastmath=True)
def insert_star_by_coords(image, star, coords):
    """
    Inserts a sample of the star onto another image at specified coordinates.

    Note: speed up with Numba

    Args:
    image (np.ndarray): The image to insert the star image into.
    star (np.ndarray): The star image to be inserted.
    coords (Tuple[int, int]): The coordinates to place the star image.

    Returns:
    np.ndarray: The image with the star image inserted.
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
        star[int(cut_top):int(star_y_size - cut_bottom), int(cut_left):int(star_x_size - cut_right)],
        image_to_correct)
    return image


@numba.jit(nopython=True, fastmath=True)
def calculate_star_form_on_single_image(image: np.ndarray, star: np.ndarray, start_coords: tuple[int, int],
                                        movement_vector: np.ndarray, exposure_time: Optional[Decimal] = None
                                        ) -> np.ndarray:
    """
    Calculate the form of a star sample on a single image based on the star sample, coordinates, movement vector,
    and exposure time. The idea is to emulate object sample movement during the single exposure.

    Note: speed up with Numba

    Args:
        image (np.ndarray): The image to insert the star form into.
        star (np.ndarray): The star sample image to be inserted.
        start_coords (Tuple[int, int]): The starting coordinates to place the star image.
        movement_vector (np.ndarray): The movement vector of the star image.
        exposure_time (Optional[Decimal], optional): The exposure time of the image. Defaults to None.

    Returns:
        np.ndarray: The image with the star form inserted.
    """
    per_image_movement_vector = movement_vector * exposure_time / 3600
    y_move, x_move = per_image_movement_vector
    start_y, start_x = start_coords
    dx = 0
    if x_move == y_move == 0:
        image = insert_star_by_coords(image, star, (start_y, start_x))
        return image
    x_moves_per_y_moves = x_move / y_move
    for dy in range(round(y_move + 1)):
        if dy * x_moves_per_y_moves // 1 > dx:
            dx += 1
        elif dy * x_moves_per_y_moves // 1 < -dx:
            dx -= 1
        image = insert_star_by_coords(image, star, (start_y + dy, start_x + dx))
        if (start_y + dy + star.shape[0] < 0 or start_y + dy - star.shape[0] > image.shape[0]
                or start_x + dx + star.shape[1] < 0 or start_x + dx - star.shape[1] > image.shape[1]):
            break
    return image


@numba.jit(nopython=True, fastmath=True)
def generate_timestamps(ts_num: int) -> tuple[list[int], float]:
    """
    Generate random timestamps emulating timestamps of images. It's needed to emulate different number of sessions and
    different exposure times within each session.

    Args:
        ts_num (int): The number of timestamps to generate.

    Returns:
        Tuple: A tuple containing a list of generated timestamps and a numpy array of random exposure times.
    """
    # generate random timestamps emulating timestamps of observations sessions staring from now
    # each session should have at least 5 timestamps. If there are less than 5 timestamps - only one session will be
    # generated. number of sessions is random. Interval between timestamps is equal to the same random exposure time
    # applicable for all intervals plus some random offset
    min_frames_per_session = 8

    max_sessions_num = min((ts_num - 1) // 8, 4)
    sessions_num = 0 if max_sessions_num == 0 else random.randrange(0, max_sessions_num)
    exposures = np.array([0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    exposure = float(exposures[random.randrange(len(exposures))])

    if sessions_num == 0:
        frames_per_session = [ts_num]
    else:
        frames_per_session = []
        remaining_sum = ts_num
        for i in range(sessions_num):
            num = random.randint(min_frames_per_session, remaining_sum - min_frames_per_session * (sessions_num - i))
            frames_per_session.append(num)
            remaining_sum -= num
        frames_per_session.append(remaining_sum)

    new_session_starts = []
    for i in range(len(frames_per_session) - 1):
        new_session_starts.append(sum(frames_per_session[:i + 1]))
    start_ts = 0
    timestamps = []
    max_inter_exposure = 20 * 60  # 20 minutes
    added_days = 0
    for num in range(ts_num):
        if num == 0:
            next_timestamp = 0
        elif num in new_session_starts:
            added_days += 1
            next_timestamp = start_ts + added_days * 24 * 3600 + random.randrange(
                0, 3) * 60 + random.randrange(0, 3600)
        else:
            next_timestamp = timestamps[-1] + exposure + random.randrange(1, max_inter_exposure + 1)
        timestamps.append(next_timestamp)
    return timestamps, exposure


@numba.jit(nopython=True, fastmath=True)
def generate_random_objects(imgs_num: int, star_samples: list[np.ndarray]
                            ) -> tuple[int, int, int, np.ndarray, float, np.ndarray, float, list[int]]:
    """
    Generates random objects based on the provided image number and star samples.

    Args:
        imgs_num (int): The number of images.
        star_samples: The samples of stars for object generation.

    Returns:
        Tuple: A tuple containing random object properties.
    """
    start_y = random.randrange(0, CHUNK_SIZE)
    start_x = random.randrange(0, CHUNK_SIZE)
    start_frame_idx = random.randrange(0, imgs_num)
    timestamps, exposure = generate_timestamps(imgs_num)
    brightness_above_noize = float(random.randrange(500, 1001)) / 1000
    star_sample = star_samples[random.randrange(len(star_samples))]
    total_time = timestamps[-1] - timestamps[0]
    total_time /= 3600
    min_vector = max(MIN_TOTAL_MOVEMENT / total_time, 0.5)
    max_vector = 30.  # pixels/hour
    vector_len = random.uniform(min_vector, max_vector)
    movement_angle = random.uniform(0., 2 * np.pi)
    movement_vector = np.array([np.sin(movement_angle), np.cos(movement_angle)], dtype=np.float32) * vector_len
    return start_y, start_x, start_frame_idx, movement_vector, brightness_above_noize, star_sample, exposure, timestamps


if __name__ == '__main__':
    pass

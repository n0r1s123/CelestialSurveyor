import json
import numpy as np
import os
import requests
import sys
import uuid

from astropy.wcs import WCS
from astropy.coordinates import Angle
from threading import Event
from typing import Optional, Union, Tuple, Generator

from backend.consuming_functions.load_headers import load_headers
from backend.consuming_functions.load_images import load_images, PIXEL_TYPE, load_image
from backend.consuming_functions.plate_solve_images import plate_solve, plate_solve_image
from backend.consuming_functions.align_images import align_images_wcs
from backend.consuming_functions.stretch_images import stretch_images
from backend.data_classes import SharedMemoryParams
from backend.progress_bar import AbstractProgressBar
from backend.data_classes import Header
from backend.known_object import KnownObject
from logger.logger import get_logger


logger = get_logger()


CHUNK_SIZE = 64


class SourceDataV2:
    """
    Class to manage image data.
    """
    def __init__(self, to_debayer: bool = False) -> None:
        self.headers = []
        self.original_frames = None
        self.shm = None
        if not os.path.exists(self.tmp_folder):
            os.mkdir(self.tmp_folder)
        else:
            # only one sourcedata instance to be loaded at the same time
            for item in os.listdir(self.tmp_folder):
                if item.endswith(".np"):
                    os.remove(os.path.join(self.tmp_folder, item))

        self.shm_name = self.__create_shm_name('images')
        self.shm_params = None
        self.footprint_map = None
        self.to_debayer = to_debayer
        self.y_borders: slice = slice(None, None)
        self.x_borders: slice = slice(None, None)
        self.__usage_map = None
        self.__chunk_len: int = 0
        self.__wcs: Optional[WCS] = None
        self.__cropped = False
        self.__shared = True
        self.__images = None
        self.__stop_event = Event()
        self.__used_images = None
        self.__usage_map_changed = True
        self.__original_frames = None

    def __create_shm_name(self, postfix: str = '') -> str:
        """
        Creates name for the shared memory file.

        Parameters:
        - postfix (str): Optional postfix to append to the shared memory file name.

        Returns:
        - str: The generated shared memory file name.
        """
        shm_name = os.path.join(self.tmp_folder, f"tmp_{uuid.uuid4().hex}_{postfix}.np")
        return shm_name

    def __clear_tmp_folder(self):
        """
        Clears temporary folder by removing all files with '.np' extension (shared memory files).
        """
        for item in os.listdir(self.tmp_folder):
            if item.endswith(".np"):
                os.remove(os.path.join(self.tmp_folder, item))

    def __reset_shm(self):
        """
        Resets the shared memory by clearing the temporary folder and creating a new shared memory file.
        Required in UI mode when user wants to add more images or stops loading data.
        """
        self.original_frames = None
        self.__clear_tmp_folder()
        self.shm_name = self.__create_shm_name('images')

    def raise_stop_event(self):
        """
        Raise the stop event to let the child processes to stop.
        """
        self.__stop_event.set()

    def clear_stop_event(self):
        """
        Raise the stop event to let the child processes that reloading may be done.
        """
        self.__stop_event.clear()

    @property
    def tmp_folder(self) -> str:
        """
        Get the path to the temporary folder where shared memory files are stored.

        Returns:
            str: The path to the temporary folder.
        """
        return os.path.join(sys.path[1], "tmp")

    @property
    def stop_event(self) -> Event:
        return self.__stop_event

    @staticmethod
    def filter_file_list(file_list: list[str]) -> list[str]:
        """
        Filter the file list to include only files with extensions .xisf, .fit, or .fits.

        Args:
            file_list (list[str]): List of file paths to filter.

        Returns:
            list[str]: Filtered list of file paths.
        """
        return [item for item in file_list if item.lower().endswith(".xisf") or item.lower().endswith(".fit")
                or item.lower().endswith(".fits")]

    def extend_headers(self, file_list: list[str], progress_bar: Optional[AbstractProgressBar] = None) -> None:
        """
        Extends the headers with information loaded from the given file list.

        Args:
            file_list (list[str]): List of file paths to load headers from.
            progress_bar (Optional[AbstractProgressBar]): An optional progress bar to show the loading progress.

        Returns:
            None
        """
        file_list = self.filter_file_list(file_list)
        self.headers.extend(load_headers(file_list, progress_bar, stop_event=self.stop_event))
        self.headers.sort(key=lambda header: header.timestamp)

    def set_headers(self, headers: list[Header]) -> None:
        """
        Set the headers.

        Args:
            headers (list[Header]): List of headers to set.

        Returns:
            None
        """
        self.headers = headers
        self.headers.sort(key=lambda header: header.timestamp)
        self.__reset_shm()

    @property
    def shape(self) -> tuple:
        return self.images.shape

    @property
    def origional_shape(self) -> tuple:
        return self.original_frames.shape

    @property
    def usage_map(self) -> np.ndarray:
        if self.__usage_map is None:
            self.__usage_map = np.ones((len(self.__images), ), dtype=bool)
        return self.__usage_map

    @usage_map.setter
    def usage_map(self, usage_map):
        self.__usage_map = usage_map
        self.__usage_map_changed = True

    @property
    def images(self):
        if self.__shared:
            usage_map = self.__usage_map if self.__usage_map is not None else np.ones((len(self.headers), ), dtype=bool)
            return self.original_frames[
                usage_map, self.y_borders, self.x_borders] if self.original_frames is not None else None
        else:
            if self.__usage_map_changed:
                self.__used_images = self.__images[self.usage_map]
                self.__usage_map_changed = False
            return self.__used_images

    def images_from_buffer(self) -> None:
        """
        Copy images from the shared memory file to RAM, update headers based on usage map,
        and reset shared memory properties. Needs to be done after image loading, calibration and alignment to speed up
        processing.
        """
        self.__images = np.copy(self.images)
        self.headers = [header for idx, header in enumerate(self.headers) if self.usage_map[idx]]
        self.__usage_map_changed = True
        self.usage_map = np.ones((len(self.__images), ), dtype=bool)
        name = self.shm_name
        self.original_frames._mmap.close()
        del self.original_frames
        self.original_frames = None
        os.remove(name)
        self.__original_frames = None
        self.__shared = False

    @property
    def max_image(self) -> np.ndarray:
        return np.amax(self.images, axis=0)

    @property
    def wcs(self) -> WCS:
        if self.__wcs is None and self.__cropped is True:
            self.__wcs, _ = self.plate_solve()
        return self.__wcs

    @wcs.setter
    def wcs(self, value):
        self.__wcs = value

    def load_images(self, progress_bar: Optional[AbstractProgressBar] = None) -> None:
        """
        Load images from the file list specified in the headers.

        Parameters:
        - progress_bar (Optional[AbstractProgressBar]): A progress bar to show the loading progress.

        Returns:
        - None
        """
        logger.log.info("Loading images...")
        file_list = [header.file_name for header in self.headers]
        img = load_image(file_list[0])
        shape = (len(file_list), *img.shape)
        self.shm_params = SharedMemoryParams(
            shm_name=self.shm_name, shm_shape=shape, shm_size=img.nbytes * len(file_list), shm_dtype=img.dtype)
        self.original_frames = np.memmap(self.shm_name, dtype=PIXEL_TYPE, mode='w+', shape=shape)
        self.__shared = True
        load_images(
            file_list, self.shm_params, to_debayer=self.to_debayer, progress_bar=progress_bar,
            stop_event=self.stop_event)

    @staticmethod
    def calculate_raw_crop(footprint: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Calculate the crop coordinates based on the footprint array.
        Raw crop means that the lines and columns which contain only zeros will be cut.

        Parameters:
        - footprint (np.ndarray): The input array representing the footprint.

        Returns:
        - Tuple[Tuple[int, int], Tuple[int, int]]: A tuple containing the crop coordinates for y-axis and x-axis.

        """
        y_top = x_left = 0
        y_bottom, x_right = footprint.shape[:2]
        for num, line in enumerate(footprint):
            if not np.all(line):
                y_top = num
                break
        for num, line in enumerate(footprint[::-1]):
            if not np.all(line):
                y_bottom -= num
                break

        for num, line in enumerate(footprint.T):
            if not np.all(line):
                x_left = num
                break

        for num, line in enumerate(footprint.T[::-1]):
            if not np.all(line):
                x_right -= num
                break

        return (y_top, y_bottom), (x_left, x_right)

    @staticmethod
    def crop_image(imgs: np.ndarray,
                   y_borders: Union[slice, Tuple[int, int]],
                   x_borders: Union[slice, Tuple[int, int]],
                   usage_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Crop the images based on the provided y and x borders.

        Args:
            imgs (np.ndarray): The input image array.
            y_borders (Union[slice, Tuple[int, int]]): The borders for the y-axis.
            x_borders (Union[slice, Tuple[int, int]]): The borders for the x-axis.
            usage_mask (Optional[np.ndarray]): Optional usage mask for cropping.

        Returns:
            np.ndarray: The cropped image array.
        """
        if isinstance(y_borders, slice):
            pass
        elif isinstance(y_borders, tuple) and len(y_borders) == 2:
            y_borders = slice(*y_borders)
        else:
            raise ValueError("y_borders must be a tuple of length 2 or a slice")
        if isinstance(x_borders, slice):
            pass
        elif isinstance(x_borders, tuple) and len(x_borders) == 2:
            x_borders = slice(*x_borders)
        else:
            raise ValueError("x_borders must be a tuple of length 2 or a slice")
        if usage_mask:
            return imgs[usage_mask, y_borders, x_borders]
        else:
            return imgs[:, y_borders, x_borders]

    @staticmethod
    def __get_num_of_corner_zeros(line: np.ndarray) -> int:
        """
        Count the number of zeros in the footprint line.

        Args:
            line (np.ndarray): The input line from the footprint.

        Returns:
            int: The number of zeros in the line.
        """
        # True means zero in footprint
        return np.count_nonzero(line)

    @classmethod
    def __fine_crop_border(cls, footprint: np.ndarray, direction: int, transpon: bool = True) -> Tuple[np.array, int]:
        """
        This method calculates the fine crop border based on the direction and whether to transpose the footprint.
        The goal is to leave areas where the image is without zeros after alignment.

        Args:
            footprint (np.ndarray): The input footprint array.
            direction (int): The direction to calculate the border.
            transpon (bool, optional): Whether to transpose the footprint. Defaults to True.

        Returns:
            Tuple[np.array, int]: The cropped footprint and the calculated border.
        """
        if transpon:
            footprint = footprint.T
        x = 0
        line: np.ndarray
        for num, line in enumerate(footprint[::direction]):
            if cls.__get_num_of_corner_zeros(line) <= cls.__get_num_of_corner_zeros(
                    footprint[::direction][num + 1]):
                x = num
                break
        if direction == -1:
            result_tmp = footprint[: (x + 1) * direction]
            x = footprint.shape[0] - x
        else:
            result_tmp = footprint[x:]
        return result_tmp.T if transpon else result_tmp, x

    @classmethod
    def calculate_crop(cls, footprint: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Calculate the crop coordinates based on the footprint array. The goal is to calculate rectangle without zero
        areas after alignment.

        Parameters:
        - footprint (np.ndarray): The input array representing the footprint.

        Returns:
        - Tuple[Tuple[int, int], Tuple[int, int]]: A tuple containing the crop coordinates for y-axis and x-axis.
        """

        y_pre_crop, x_pre_crop = cls.calculate_raw_crop(footprint)
        pre_cropped = footprint[slice(*y_pre_crop), slice(*x_pre_crop)]
        y_top_zeros = cls.__get_num_of_corner_zeros(pre_cropped[0])
        y_bottom_zeros = cls.__get_num_of_corner_zeros(pre_cropped[-1])
        x_left_zeros = cls.__get_num_of_corner_zeros(pre_cropped.T[0])
        x_right_zeros = cls.__get_num_of_corner_zeros(pre_cropped.T[-1])
        zeros = y_top_zeros, y_bottom_zeros, x_left_zeros, x_right_zeros
        trim_args = (1, False), (-1, False), (1, True), (-1, True)
        args_order = (item[1] for item in sorted(zip(zeros, trim_args), key=lambda x: x[0], reverse=True))
        border_map = {item: value for item, value in zip(trim_args, ["y_top", "y_bottom", "x_left", "x_right"])}
        result = {}
        cropped = pre_cropped
        for pair in args_order:
            boarder_name = border_map[pair]
            cropped, x = cls.__fine_crop_border(cropped, *pair)
            result.update({boarder_name: x})

        y_top = result["y_top"] + y_pre_crop[0]
        y_bottom = result["y_bottom"] + y_pre_crop[0]
        x_left = result["x_left"] + x_pre_crop[0]
        x_right = result["x_right"] + x_pre_crop[0]
        crop = (y_top, y_bottom), (x_left, x_right)
        return crop

    def align_images_wcs(self, progress_bar: Optional[AbstractProgressBar] = None) -> None:
        """
        Align images with World Coordinate System (WCS).

        Args:
            progress_bar (Optional[AbstractProgressBar]): Progress bar object.

        Returns:
            None
        """
        logger.log.info("Aligning images...")
        success_map, self.footprint_map = align_images_wcs(
            self.shm_params,
            [header.wcs for header in self.headers],
            progress_bar=progress_bar,
            stop_event=self.stop_event)
        self.__usage_map = success_map

    def crop_images(self) -> None:
        """
        Crop the images based on the footprint map and update borders.
        Keeps common non-zero area on all the images after alignment.
        """
        logger.log.info("Cropping images...")
        x_borders, y_borders = [], []
        if self.footprint_map is None:
            footprint_map = self.original_frames[self.__usage_map] == 0
            footprint_map = footprint_map[0]
            if len(footprint_map.shape) == 4:
                footprint_map = np.reshape(footprint_map, footprint_map.shape[:-1])
        else:
            footprint_map = self.footprint_map[np.array(self.__usage_map, dtype=bool)]
        for item in footprint_map:
            if self.stop_event.is_set():
                return
            y_border, x_border = self.calculate_crop(item)

            y_borders.append(y_border)
            x_borders.append(x_border)
        y_borders = np.array(y_borders)
        x_borders = np.array(x_borders)
        self.y_borders = slice(int(np.max(y_borders[:, 0])), int(np.min(y_borders[:, 1])))
        self.x_borders = slice(int(np.max(x_borders[:, 0])), int(np.min(x_borders[:, 1])))
        self.__cropped = True
        self.footprint_map = None
        # Plate solve after cropping
        self.wcs, _ = self.plate_solve(0)

    def make_master_dark(self, filenames: list[str], progress_bar: Optional[AbstractProgressBar] = None) -> np.ndarray:
        """
        Create a master dark frame from a list of dark frame filenames.

        Args:
            filenames (list[str]): List of dark frame filenames.
            progress_bar (Optional[AbstractProgressBar], optional): Progress bar instance. Defaults to None.

        Returns:
            np.ndarray: Master dark frame.
        """
        shape = (len(filenames), *self.origional_shape[1:])
        size = self.original_frames.itemsize
        shm_name = self.__create_shm_name('darks')
        for value in shape:
            size *= value
        shm_params = SharedMemoryParams(
            shm_name=shm_name, shm_shape=shape, shm_size=size, shm_dtype=PIXEL_TYPE)
        darks = np.memmap(shm_params.shm_name, dtype=PIXEL_TYPE, mode='w+', shape=shape)
        load_images(filenames, shm_params, progress_bar=progress_bar, to_debayer=self.to_debayer)
        master_dark = np.average(darks, axis=0)
        darks._mmap.close()
        del darks
        os.remove(shm_name)
        return master_dark

    def make_master_flat(self, flat_filenames: list[str], dark_flat_filenames: Optional[list[str]] = None,
                         progress_bar: Optional[AbstractProgressBar] = None) -> np.ndarray:
        """
        Create a master flat frame from a list of flat frame filenames.

        Args:
            flat_filenames (list[str]): List of flat frame filenames.
            dark_flat_filenames (Optional[list[str]], optional): List of dark flat frame filenames. Defaults to None.
            progress_bar (Optional[AbstractProgressBar], optional): Progress bar instance. Defaults to None.

        Returns:
            np.ndarray: Master flat frame.
        """
        flat_shape = (len(flat_filenames), *self.origional_shape[1:])
        flat_size = self.original_frames.itemsize
        flat_shm_name = self.__create_shm_name('flats')
        for value in flat_shape:
            flat_size *= value
        flat_shm_params = SharedMemoryParams(
            shm_name=flat_shm_name, shm_shape=flat_shape, shm_size=flat_size, shm_dtype=PIXEL_TYPE)
        flats = np.memmap(flat_shm_params.shm_name, dtype=PIXEL_TYPE, mode='w+', shape=flat_shape)
        load_images(flat_filenames, flat_shm_params, progress_bar=progress_bar, to_debayer=self.to_debayer)
        if dark_flat_filenames is not None:
            master_dark_flat = self.make_master_dark(dark_flat_filenames, progress_bar=progress_bar)
            for flat in flats:
                flat -= master_dark_flat
        master_flat = np.average(flats, axis=0)
        flats._mmap.close()
        del flats
        os.remove(flat_shm_name)
        return master_flat

    def load_flats(self, flat_filenames: list[str], progress_bar: Optional[AbstractProgressBar] = None) -> np.ndarray:
        """
        Load flat frames into memory.

        Args:
            flat_filenames (list[str]): List of flat frame filenames.
            progress_bar (Optional[AbstractProgressBar], optional): Progress bar instance. Defaults to None.

        Returns:
            np.ndarray: Loaded flat frames.
        """
        flat_shape = (len(flat_filenames), *self.origional_shape[1:])
        flat_size = self.original_frames.itemsize
        flat_shm_name = self.__create_shm_name('flats')
        for value in flat_shape:
            flat_size *= value
        flat_shm_params = SharedMemoryParams(
            shm_name=flat_shm_name, shm_shape=flat_shape, shm_size=flat_size, shm_dtype=PIXEL_TYPE)
        flats = np.memmap(flat_shm_params.shm_name, dtype=PIXEL_TYPE, mode='w+', shape=flat_shape)
        load_images(flat_filenames, flat_shm_params, progress_bar=progress_bar, to_debayer=self.to_debayer)
        res = np.copy(flats)
        flats._mmap.close()
        del flats
        os.remove(flat_shm_name)
        return res

    def calibrate_images(self, dark_files: Optional[list[str]] = None, flat_files: Optional[list[str]] = None,
                         dark_flat_files: Optional[list[str]] = None, progress_bar: Optional[AbstractProgressBar] = None
                         ) -> None:
        """
        Calibrates images by subtracting master dark frames and dividing by master flat frames.

        Args:
            dark_files (Optional[list[str]]): List of dark frame filenames.
            flat_files (Optional[list[str]]): List of flat frame filenames.
            dark_flat_files (Optional[list[str]]): List of dark flat frame filenames.
            progress_bar (Optional[AbstractProgressBar]): Progress bar instance.

        Returns:
            None
        """
        if dark_files is not None:
            master_dark = self.make_master_dark(dark_files, progress_bar=progress_bar)
            self.original_frames -= master_dark
        if flat_files is not None:
            master_flat = self.make_master_flat(flat_files, dark_flat_files, progress_bar=progress_bar)
            self.original_frames /= master_flat

    def stretch_images(self, progress_bar: Optional[AbstractProgressBar] = None) -> None:
        """
        Stretch images stored in shared memory.

        Args:
            progress_bar (Optional[AbstractProgressBar]): Progress bar to track the stretching progress.

        Returns:
            None
        """
        logger.log.info("Stretching images...")
        shm_params = self.shm_params
        shm_params.y_slice = self.y_borders
        shm_params.x_slice = self.x_borders
        stretch_images(self.shm_params, progress_bar=progress_bar, stop_event=self.stop_event)

    def get_number_of_chunks(self, size: tuple[int, int] = (CHUNK_SIZE, CHUNK_SIZE),
                             overlap: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the number of image chunks based on the specified size and overlap.

        Args:
            size (tuple[int, int], optional): The size of the image chunks in the format (height, width).
                Defaults to (CHUNK_SIZE, CHUNK_SIZE).
            overlap (float, optional): The overlap percentage between image chunks. Defaults to 0.5.

        Returns:
            tuple[np.ndarray, np.ndarray]: Two arrays representing the y and x coordinates of the image chunks.
        """
        size_y, size_x = size
        ys = np.arange(0, self.shape[1] - size_y * overlap, size_y * overlap)
        ys[-1] = self.shape[1] - size_y
        xs = np.arange(0, self.shape[2] - size_x * overlap, size_x * overlap)
        xs[-1] = self.shape[2] - size_x
        return ys, xs

    def generate_image_chunks(self, size: tuple[int, int] = (CHUNK_SIZE, CHUNK_SIZE), overlap: float = 0.5):
        """
        Generate image chunks based on the specified size and overlap.

        Args:
            size (tuple[int, int], optional): The size of the image chunks in the format (height, width).
                Defaults to (CHUNK_SIZE, CHUNK_SIZE).
            overlap (float, optional): The overlap percentage between image chunks. Defaults to 0.5.

        Yields:
            tuple: A tuple containing the coordinates and prepared images of the generated image chunks.
        """
        size_y, size_x = size
        ys, xs = self.get_number_of_chunks(size, overlap)
        coordinates = ((y, x) for y in ys for x in xs)
        for y, x in coordinates:
            y, x = int(y), int(x)
            imgs = np.copy(self.images[:, y:y + size_y, x:x + size_x])
            yield (y, x), self.prepare_images(np.copy(imgs))

    @staticmethod
    def generate_batch(chunk_generator: Generator, batch_size: int) -> tuple[tuple[int, int], np.ndarray]:
        """
        Generate batches of chunks for processing with the given batch size.

        Args:
            chunk_generator (Generator): Generator that yields chunks and coordinates.
            batch_size (int): The size of each batch.

        Yields:
            tuple[tuple[int, int], np.ndarray]: A tuple containing the coordinates and batch of chunks.
        """
        batch = []
        coords = []
        for coord, chunk in chunk_generator:
            batch.append(chunk)
            coords.append(coord)
            if len(batch) == batch_size:
                yield coords, np.array(batch)
                batch = []
                coords = []
        if len(batch) > 0:
            yield coords, np.array(batch)

    @staticmethod
    def estimate_image_noize_level(imgs: np.ndarray) -> float:
        """
        Estimate the noise level of the given images.

        Args:
            imgs (np.ndarray): The images to estimate the noise level for.

        Returns:
            float: The estimated noise level.
        """
        return np.mean(np.var(imgs, axis=0))

    @classmethod
    def prepare_images(cls, images: np.ndarray) -> np.ndarray:
        """
        Prepare the given images for processing by AI model.

        Args:
            images (np.ndarray): The images to prepare.

        Returns:
            np.ndarray: The prepared images.
        """
        # normalize images
        images -= cls.estimate_image_noize_level(images)
        images = images - np.min(images)
        images = images / np.max(images)
        images = np.reshape(images, (*images.shape[:3], 1))
        return images

    @staticmethod
    def adjust_chunks_to_min_len(imgs: np.ndarray, timestamps: list, min_len: int = 8) -> tuple[np.ndarray, list]:
        """
        Adjust the given chunks to the minimum length. To be used if there are fewer images than the minimum length.

        Args:
            imgs (np.ndarray): The images to adjust.
            timestamps (list): The timestamps of the images.
            min_len (int, optional): The minimum length of the chunks. Defaults to 8.

        Returns:
            tuple[np.ndarray, list]: The adjusted images and timestamps.
        """
        assert len(imgs) == len(timestamps), \
            f"Images and timestamp amount mismatch: len(imgs)={len(imgs)}. len(timestamps)={len(timestamps)}"

        if len(imgs) >= min_len:
            return imgs, timestamps
        new_imgs = []
        new_timestamps = []
        while len(new_imgs) < min_len:
            new_imgs.extend(list(imgs))
            new_timestamps.extend(timestamps)
        new_imgs = new_imgs[:8]
        new_timestamps = new_timestamps[:8]
        timestamped_images = list(zip(new_imgs, new_timestamps))
        timestamped_images.sort(key=lambda x: x[1])
        new_imgs = [item[0] for item in timestamped_images]
        new_timestamps = [item[1] for item in timestamped_images]
        new_imgs = np.array(new_imgs)
        return new_imgs, new_timestamps

    @staticmethod
    def make_file_paths(folder: str) -> list[str]:
        """
        Create a list of file paths for files in the specified folder that end with specific extensions.

        Parameters:
        - folder (str): The folder path to search for files.

        Returns:
        - list[str]: A list of file paths with extensions '.xisf', '.fit', or '.fits'.
        """
        return [os.path.join(folder, item) for item in os.listdir(folder) if item.lower().endswith(
            ".xisf") or item.lower().endswith(".fit") or item.lower().endswith(".fits")]

    def plate_solve(self, ref_idx: int = 0, sky_coord: Optional[np.ndarray] = None) -> tuple[WCS, np.ndarray]:
        """
        Perform plate solving on the given reference image.

        Args:
            ref_idx (int, optional): The index of the reference image. Defaults to 0.
            sky_coord (np.ndarray, optional): The sky coordinates of the reference image. Defaults to None.

        Returns:
            tuple[WCS, np.ndarray]: The plate solved WCS and the sky coordinates of the reference image.
        """
        logger.log.info("Plate solving...")
        wcs, sky_coord = plate_solve_image(self.images[ref_idx], header=self.headers[ref_idx], sky_coord=sky_coord)
        self.__wcs = wcs
        return wcs, sky_coord

    def plate_solve_all(self, progress_bar: Optional[AbstractProgressBar] = None) -> None:
        """
        Perform plate solving on all images and update corresponding headers.

        Args:
            progress_bar (AbstractProgressBar, optional): The progress bar to display the progress. Defaults to None.

        Returns:
            None
        """
        logger.log.info("Plate solving all images...")
        res = plate_solve(self.shm_params, self.headers, progress_bar=progress_bar, stop_event=self.stop_event)
        for wcs, header in zip(res, self.headers):
            header.wcs = wcs

    @staticmethod
    def convert_ra(ra: Angle) -> str:
        """
        Convert right ascension from astropy Angle format to a string representation suitable for Small Body Api.
        https://ssd-api.jpl.nasa.gov/doc/sb_ident.html

        Args:
            ra (Angle): The right ascension angle to convert.

        Returns:
            str: The string representation of the right ascension.
        """
        minus_substr = "M" if int(ra.h) < 0 else ""
        hour = f"{minus_substr}{abs(int(ra.h)):02d}"
        return f"{hour}-{abs(int(ra.m)):02d}-{abs(int(ra.s)):02d}"

    @staticmethod
    def convert_dec(dec: Angle) -> str:
        """
        Convert declination from astropy Angle format to a string representation suitable for Small Body Api.
        https://ssd-api.jpl.nasa.gov/doc/sb_ident.html

        Args:
            dec (Angle): The declination angle to convert.

        Returns:
            str: The string representation of the declination.
        """
        minus_substr = "M" if int(dec.d) < 0 else ""
        hour = f"{minus_substr}{abs(int(dec.d)):02d}"
        return f"{hour}-{abs(int(dec.m)):02d}-{abs(int(dec.s)):02d}"

    def fetch_known_asteroids_for_image(self, img_idx: int, magnitude_limit: float = 18.0
                                        ) -> tuple[list[KnownObject], list[KnownObject]]:
        """
        Fetch known asteroids and comets within the image's field of view based on the specified magnitude limit.
        Request data from JPL Small Body Api (https://ssd-api.jpl.nasa.gov/doc/sb_ident.html).

        Args:
            img_idx (int): The index of the image.
            magnitude_limit (float): The magnitude limit for known asteroids.

        Returns:
            tuple[list[KnownObject], list[KnownObject]]: A tuple containing lists of KnownObject instances for asteroids
                and comets.
        """
        logger.log.debug("Requesting visible targets")
        obs_time = self.headers[img_idx].timestamp
        obs_time = (f"{obs_time.year:04d}-{obs_time.month:02d}-{obs_time.day:02d}_{obs_time.hour:02d}:"
                    f"{obs_time.minute:02d}:{obs_time.second:02d}")
        corner_points = [self.wcs.pixel_to_world(x, y) for x, y in (
            (0, 0), (0, self.shape[1]), (self.shape[2], 0), (self.shape[2], self.shape[1]))]
        ra_max = max([item.ra.hms for item in corner_points])
        ra_min = min([item.ra.hms for item in corner_points])
        dec_max = max([item.dec.dms for item in corner_points])
        dec_min = min([item.dec.dms for item in corner_points])
        fov_ra_lim = f"{self.convert_ra(ra_min)},{self.convert_ra(ra_max)}"
        fov_dec_lim = f"{self.convert_dec(dec_min)},{self.convert_dec(dec_max)}"
        know_asteroids = []
        know_comets = []
        for sb_kind in ('a', 'c'):
            params = {
                "sb-kind": sb_kind,
                "lat": round(self.headers[img_idx].site_location.lat, 3),
                "lon": round(self.headers[img_idx].site_location.long, 4),
                "alt": 0,
                "obs-time": obs_time,
                "mag-required": True,
                "two-pass": True,
                "suppress-first-pass": True,
                "req-elem": False,
                "vmag-lim": magnitude_limit,
                "fov-ra-lim": fov_ra_lim,
                "fov-dec-lim": fov_dec_lim,
            }
            logger.log.debug(f"Params: {params}")
            res = requests.get("https://ssd-api.jpl.nasa.gov/sb_ident.api", params)
            res = json.loads(res.content)
            potential_known_objects = [dict(zip(res["fields_second"], item)) for item in
                                       res.get("data_second_pass", [])]
            potential_known_objects = [KnownObject(item, wcs=self.wcs) for item in potential_known_objects]
            first_pass_objects = [dict(zip(res["fields_first"], item)) for item in res.get("data_first_pass", [])]
            potential_known_objects.extend([KnownObject(item, wcs=self.wcs) for item in first_pass_objects])
            for item in potential_known_objects:
                x, y = item.pixel_coordinates
                if 0 <= x < self.shape[2] and 0 <= y < self.shape[1]:
                    if sb_kind == 'a':
                        know_asteroids.append(item)
                    if sb_kind == 'c':
                        know_comets.append(item)
        logger.log.info(f"Found {len(know_asteroids)} known asteroids and {len(know_comets)} known comets in the FOV")
        if know_asteroids:
            logger.log.info(f"Known asteroids:")
            for item in know_asteroids:
                logger.log.info(str(item))
        if know_comets:
            logger.log.info(f"Known comets:")
            for item in know_comets:
                logger.log.info(str(item))
        return know_asteroids, know_comets

import json
import os
import uuid
import numpy as np

from typing import Optional, Union, Tuple
from astropy.wcs import WCS
from logger.logger import get_logger

from backend.consuming_functions.load_headers import load_headers
from backend.consuming_functions.load_images import load_images, PIXEL_TYPE, load_image
from backend.consuming_functions.plate_solve_images import plate_solve, plate_solve_image
from backend.consuming_functions.align_images import align_images_wcs
from backend.consuming_functions.stretch_images import stretch_images
from backend.data_classes import SharedMemoryParams
from backend.progress_bar import ProgressBarCli, AbstractProgressBar
from backend.data_classes import Header
import requests
from backend.known_object import KnownObject
from threading import Event
from backend.consuming_functions.measure_execution_time import measure_execution_time


logger = get_logger()


CHUNK_SIZE = 64
# TMP_FOLDER = os.ro


class SourceDataV2:
    def __init__(self, to_debayer: bool = False) -> None:
        self.headers = []
        self.original_frames = None
        self.shm = None
        self.shm_name = uuid.uuid4().hex + ".np"
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

    def raise_stop_event(self):
        self.__stop_event.set()

    def clear_stop_event(self):
        self.__stop_event.clear()

    @property
    def stop_event(self):
        return self.__stop_event

    @staticmethod
    def filter_file_list(file_list: list[str]) -> list[str]:
        return [item for item in file_list if item.lower().endswith(".xisf") or item.lower().endswith(".fit")
                or item.lower().endswith(".fits")]

    def extend_headers(self, file_list: list[str], progress_bar: Optional[AbstractProgressBar] = None) -> None:
        file_list = self.filter_file_list(file_list)
        self.headers.extend(load_headers(file_list, progress_bar, stop_event=self.stop_event))
        self.headers.sort(key=lambda header: header.timestamp)

    def reload_headers(self, file_list: list[str], progress_bar: Optional[AbstractProgressBar] = None) -> None:
        file_list = self.filter_file_list(file_list)
        self.headers = load_headers(file_list, progress_bar, stop_event=self.stop_event)
        self.headers.sort(key=lambda header: header.timestamp)

    def set_headers(self, headers: list[Header]) -> None:
        self.headers = headers
        self.headers.sort(key=lambda header: header.timestamp)

    @property
    def num_frames(self) -> int:
        return len(self.headers)

    @property
    def original_shape(self):
        return self.original_frames.shape if self.original_frames is not None else None

    @property
    def shape(self):
        return self.images.shape

    @property
    def origional_shape(self):
        return self.original_frames.shape

    @property
    def usage_map(self):
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
            return self.original_frames[usage_map, self.y_borders, self.x_borders] if self.original_frames is not None else None
        else:
            if self.__usage_map_changed:
                self.__used_images = self.__images[self.usage_map]
                self.__usage_map_changed = False
            return self.__used_images

    def images_from_buffer(self):
        self.__images = np.copy(self.images)
        # images = np.zeros(self.images.shape, dtype=PIXEL_TYPE)
        # np.copyto(images, self.images)
        name = self.shm_name
        self.original_frames._mmap.close()
        del self.original_frames
        self.original_frames = None
        os.remove(name)
        self.__original_frames = None
        self.__shared = False
        # self.__images = images
        # print(type(self.images))

    @property
    def max_image(self):
        return np.amax(self.images, axis=0)

    @property
    def wcs(self):
        if self.__wcs is None and self.__cropped is True:
            self.__wcs, _ = self.plate_solve()
        return self.__wcs

    @wcs.setter
    def wcs(self, value):
        self.__wcs = value

    def load_images(self, progress_bar: Optional[AbstractProgressBar] = None) -> None:
        logger.log.info("Loading images...")
        file_list = [header.file_name for header in self.headers]
        img = load_image(file_list[0])
        shape = (len(file_list), *img.shape)
        self.shm_params = SharedMemoryParams(
            shm_name=self.shm_name, shm_shape=shape, shm_size=img.nbytes * len(file_list), shm_dtype=img.dtype)
        self.original_frames = np.memmap(self.shm_name, dtype=PIXEL_TYPE, mode='w+', shape=shape)
        self.__shared = True
        load_images(file_list, self.shm_params, to_debayer=self.to_debayer, progress_bar=progress_bar, stop_event=self.stop_event)

    @staticmethod
    def calculate_raw_crop(footprint: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
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
        # True means zero in footprint
        return np.count_nonzero(line)

    @classmethod
    def __fine_crop_border(
            cls, footprint: np.ndarray, direction: int, transpon: bool = True
    ) -> Tuple[np.array, int]:
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
        logger.log.info("Aligning images...")
        success_map, self.footprint_map = align_images_wcs(
            self.shm_params,
            [header.wcs for header in self.headers],
            progress_bar=progress_bar,
            stop_event=self.stop_event)
        self.__usage_map = success_map

    def crop_images(self):
        logger.log.info("Cropping images...")
        x_borders, y_borders = [], []
        if self.footprint_map is None:
            print(self.original_frames.shape)
            footprint_map = self.original_frames[self.__usage_map] == 0
            footprint_map = footprint_map[0]
            print(footprint_map.shape)
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
        print(self.y_borders, self.x_borders)
        self.__cropped = True
        self.footprint_map = None
        self.wcs, _ = self.plate_solve(0)

    # def secondary_align_images(self, y_splits: int = 3, x_splits: int = 3,
    #                            progress_bar: Optional[AbstractProgressBar] = None) -> None:
    #     logger.log.info(f"Secondary aligning images with {y_splits} x {x_splits} splits")
    #     y_step = (self.y_borders.stop - SECONDARY_ALIGNMENT_OFFSET - self.y_borders.start -
    #               SECONDARY_ALIGNMENT_OFFSET) // y_splits
    #     y_slices = range(self.y_borders.start + SECONDARY_ALIGNMENT_OFFSET,
    #                      self.y_borders.start + SECONDARY_ALIGNMENT_OFFSET + y_step * y_splits, y_step)
    #     y_slices = [slice(item, item + y_step) for item in y_slices]
    #     x_step = (self.x_borders.stop - SECONDARY_ALIGNMENT_OFFSET - self.x_borders.start -
    #               SECONDARY_ALIGNMENT_OFFSET) // x_splits
    #     x_slices = range(self.x_borders.start + SECONDARY_ALIGNMENT_OFFSET,
    #                      self.x_borders.start + SECONDARY_ALIGNMENT_OFFSET + x_step * x_splits, x_step)
    #     x_slices = [slice(item, item + x_step) for item in x_slices]
    #     for num_y, y_slice in enumerate(y_slices):
    #         for num_x, x_slice in enumerate(x_slices, start=1):
    #             progress_bar.clear()
    #             logger.log.info(f"Secondary aligning images part {num_y*y_splits + num_x} of {y_splits * x_splits}")
    #             shm_params = self.shm_params
    #             shm_params.y_slice = y_slice
    #             shm_params.x_slice = x_slice
    #             align_images(shm_params, progress_bar=progress_bar)
    #
    #     self.y_borders = slice(self.y_borders.start + SECONDARY_ALIGNMENT_OFFSET,
    #                            self.y_borders.start + SECONDARY_ALIGNMENT_OFFSET + y_step * y_splits)
    #     self.x_borders = slice(self.x_borders.start + SECONDARY_ALIGNMENT_OFFSET,
    #                            self.x_borders.start + SECONDARY_ALIGNMENT_OFFSET + x_step * x_splits)

    def make_master_dark(self, filenames: list[str], progress_bar: Optional[AbstractProgressBar] = None) -> np.ndarray:
        shape = (len(filenames), *self.origional_shape[1:])
        size = self.original_frames.itemsize
        shm_name = uuid.uuid4().hex + "_darks.np"
        for value in shape:
            size *= value
        # shm = SharedMemory(name=shm_name, create=True, size=size)
        shm_params = SharedMemoryParams(
            shm_name=shm_name, shm_shape=shape, shm_size=size, shm_dtype=PIXEL_TYPE)
        darks = np.memmap(shm_params.shm_name, dtype=PIXEL_TYPE, mode='w+', shape=shape)
        load_images(filenames, shm_params, progress_bar=progress_bar, to_debayer=self.to_debayer)
        # darks = np.ndarray(shape=shape, dtype=PIXEL_TYPE, buffer=shm.buf)
        master_dark = np.average(darks, axis=0)
        darks._mmap.close()
        del darks

        os.remove(shm_name)
        # shm.close()
        # shm.unlink()
        return master_dark

    def make_master_flat(self, flat_filenames: list[str], dark_flat_filenames: Optional[list[str]] = None,
                         progress_bar: Optional[AbstractProgressBar] = None) -> np.ndarray:
        flat_shape = (len(flat_filenames), *self.origional_shape[1:])
        flat_size = self.original_frames.itemsize
        flat_shm_name = uuid.uuid4().hex + "_flats.np"
        for value in flat_shape:
            flat_size *= value
        # flat_shm = SharedMemory(name=flat_shm_name, create=True, size=flat_size)
        flat_shm_params = SharedMemoryParams(
            shm_name=flat_shm_name, shm_shape=flat_shape, shm_size=flat_size, shm_dtype=PIXEL_TYPE)
        flats = np.memmap(flat_shm_params.shm_name, dtype=PIXEL_TYPE, mode='w+', shape=flat_shape)
        load_images(flat_filenames, flat_shm_params, progress_bar=progress_bar, to_debayer=self.to_debayer)
        # flats = np.ndarray(shape=flat_shape, dtype=PIXEL_TYPE, buffer=flat_shm.buf)
        if dark_flat_filenames is not None:
            master_dark_flat = self.make_master_dark(dark_flat_filenames, progress_bar=progress_bar)
            for flat in flats:
                flat -= master_dark_flat
        master_flat = np.average(flats, axis=0)
        flats._mmap.close()
        del flats
        os.remove(flat_shm_name)
        # flat_shm.close()
        # flat_shm.unlink()
        return master_flat

    def load_flats(self, flat_filenames: list[str], progress_bar: Optional[AbstractProgressBar] = None) -> np.ndarray:
        flat_shape = (len(flat_filenames), *self.origional_shape[1:])
        flat_size = self.original_frames.itemsize
        flat_shm_name = uuid.uuid4().hex + "_flats"
        for value in flat_shape:
            flat_size *= value
        # flat_shm = SharedMemory(name=flat_shm_name, create=True, size=flat_size)
        flat_shm_params = SharedMemoryParams(
            shm_name=flat_shm_name, shm_shape=flat_shape, shm_size=flat_size, shm_dtype=PIXEL_TYPE)
        flats = np.memmap(flat_shm_params.shm_name, dtype=PIXEL_TYPE, mode='w+', shape=flat_shape)
        load_images(flat_filenames, flat_shm_params, progress_bar=progress_bar, to_debayer=self.to_debayer)
        # flats = np.copy(np.ndarray(shape=flat_shape, dtype=PIXEL_TYPE, buffer=flat_shm.buf))
        # flat_shm.close()
        # flat_shm.unlink()
        res = np.copy(flats)
        flats._mmap.close()
        del flats
        os.remove(flat_shm_name)
        return res

    def calibrate_images(self, dark_files: Optional[list[str]] = None, flat_files: Optional[list[str]] = None,
                         dark_flat_files: Optional[list[str]] = None, progress_bar: Optional[AbstractProgressBar] = None
                         ) -> None:
        if dark_files is not None:
            master_dark = self.make_master_dark(dark_files, progress_bar=progress_bar)
            self.original_frames -= master_dark
        if flat_files is not None:
            master_flat = self.make_master_flat(flat_files, dark_flat_files, progress_bar=progress_bar)
            self.original_frames /= master_flat

    def stretch_images(self, progress_bar: Optional[AbstractProgressBar] = None) -> None:
        logger.log.info("Stretching images...")
        shm_params = self.shm_params
        shm_params.y_slice = self.y_borders
        shm_params.x_slice = self.x_borders
        stretch_images(self.shm_params, progress_bar=progress_bar, stop_event=self.stop_event)

    def get_number_of_chunks(self, size: tuple[int, int] = (CHUNK_SIZE, CHUNK_SIZE), overlap=0.5) -> tuple[np.ndarray, np.ndarray]:
        size_y, size_x = size
        ys = np.arange(0, self.shape[1] - size_y * overlap, size_y * overlap)
        ys[-1] = self.shape[1] - size_y
        xs = np.arange(0, self.shape[2] - size_x * overlap, size_x * overlap)
        xs[-1] = self.shape[2] - size_x
        return ys, xs

    def generate_image_chunks(self, size: tuple[int, int] = (CHUNK_SIZE, CHUNK_SIZE), overlap=0.5):
        size_y, size_x = size
        ys, xs = self.get_number_of_chunks(size, overlap)
        coordinates = ((y, x) for y in ys for x in xs)
        for y, x in coordinates:
            y, x = int(y), int(x)
            imgs = np.copy(self.images[:, y:y + size_y, x:x + size_x])
            yield (y, x), self.prepare_images(np.copy(imgs))

    def generate_batch(self, chunk_generator, batch_size: int):
        batch = []
        coords = []
        # slow_batch = []
        # for coord, chunk, slow_chunk in chunk_generator:
        for coord, chunk in chunk_generator:
            batch.append(chunk)
            coords.append(coord)
            # slow_batch.append(slow_chunk)
            if len(batch) == batch_size:
                # yield coords, np.array(batch), np.array(slow_batch)
                yield coords, np.array(batch)
                batch = []
                coords = []
                # slow_batch = []
        if len(batch) > 0:
            # return last portion of chunks. last batch may be less than batch_size
            # yield coords, np.array(batch), np.array(slow_batch)
            yield coords, np.array(batch)

    @staticmethod
    def estimate_image_noize_level(imgs):
        return np.mean(np.var(imgs, axis=0))

    @classmethod
    def prepare_images(cls, images):
        # normalize images
        images -= cls.estimate_image_noize_level(images)
        images = images - np.min(images)
        images = images / np.max(images)
        images = np.reshape(images, (*images.shape[:3], 1))
        return images

    @staticmethod
    def adjust_chunks_to_min_len(imgs, timestamps, min_len=8):
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
    def make_file_list(folder: str) -> list[str]:
        file_list = os.listdir(folder)
        file_list = [os.path.join(folder, item) for item in file_list if item.lower().endswith(".xisf") or
                     item.lower().endswith(".fit") or item.lower().endswith(".fits")][:50]
        return file_list

    @staticmethod
    def make_file_paths(folder):
        return [os.path.join(folder, item) for item in os.listdir(folder) if item.lower().endswith(
            ".xisf") or item.lower().endswith(".fit") or item.lower().endswith(".fits")]

    def plate_solve(self, ref_idx: int = 0, sky_coord: Optional[np.ndarray] = None):
        logger.log.info("Plate solving...")
        wcs, sky_coord = plate_solve_image(self.images[ref_idx], header=self.headers[ref_idx], sky_coord=sky_coord)
        self.__wcs = wcs
        return wcs, sky_coord

    def plate_solve_all(self, progress_bar: Optional[AbstractProgressBar] = None):
        res = plate_solve(self.shm_params, self.headers, progress_bar=progress_bar, stop_event=self.stop_event)
        for wcs, header in zip(res, self.headers):
            header.wcs = wcs

    @staticmethod
    def convert_ra(ra):
        minus_substr = "M" if int(ra.h) < 0 else ""
        hour = f"{minus_substr}{abs(int(ra.h)):02d}"
        return f"{hour}-{abs(int(ra.m)):02d}-{abs(int(ra.s)):02d}"

    @staticmethod
    def convert_dec(dec):
        minus_substr = "M" if int(dec.d) < 0 else ""
        hour = f"{minus_substr}{abs(int(dec.d)):02d}"
        return f"{hour}-{abs(int(dec.m)):02d}-{abs(int(dec.s)):02d}"

    def fetch_known_asteroids_for_image(self, img_idx: int, magnitude_limit: float = 18.0):
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

        # TODO: magnitude handling
        # magnitude_limit = 19
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
            print(json.dumps(res, indent=4))
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

        return know_asteroids, know_comets

    @staticmethod
    def make_file_list(folder: str) -> list[str]:
        file_list = os.listdir(folder)
        file_list = [os.path.join(folder, item) for item in file_list if item.lower().endswith(".xisf") or
                     item.lower().endswith(".fit") or item.lower().endswith(".fits")][:50]
        return file_list


if __name__ == '__main__':
    folder = "D:\\git\\dataset\\Virgo"
    file_list = SourceDataV2.make_file_list(folder)
    source_data = SourceDataV2(file_list, to_debayer=True)
    source_data.load_images(progress_bar=ProgressBarCli())
    # for img in source_data.original_frames:
    print(source_data.estimate_image_noize_level(source_data.original_frames))
    print(np.median(source_data.original_frames))

    # dark_folder = "E:\\Astro\\NGC2264\\Dark_600"
    # dark_list = SourceDataV2.make_file_list(dark_folder)
    #
    # flat_folder = "E:\\Astro\\NGC2264\\Flat"
    # dark_flat_folder = "E:\\Astro\\NGC2264\\DarkFlat"
    # flat_list = SourceDataV2.make_file_list(flat_folder)
    # dark_flat_list = SourceDataV2.make_file_list(dark_flat_folder)
    # source_data.calibrate_images(dark_list, flat_list, dark_flat_list, progress_bar=ProgressBarCli())
    #
    # source_data.align_images(progress_bar=ProgressBarCli())
    # source_data.crop_images()
    #
    # source_data.secondary_align_images(y_splits=3, x_splits=3, progress_bar=ProgressBarCli())
    # source_data.stretch_images(progress_bar=ProgressBarCli())
    # asyncio.run(source_data.get_known_asteroids())

    # gen = source_data.generate_image_chunks((256, 256))
    # for item in next(gen):
    #     img = item * (256 * 256 - 1)
    #     img = img.astype(np.uint16)
    #     img = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3))
    #     cv2.imshow("img", img)
    #     cv2.waitKey(0)



    # img = source_data.max_image * (256 * 256 - 1)
    # img = img.astype(np.uint16)
    # img = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3))
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

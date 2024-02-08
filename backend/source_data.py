import copy
import sys

import json
import traceback
import datetime
import os
import re
import wx
from typing import Optional

import astropy.io.fits
import time

import numpy as np
import tqdm
import decimal

from auto_stretch.stretch import Stretch
from xisf import XISF
from bs4 import BeautifulSoup
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory
import astroalign as aa
from logger.logger import get_logger, arg_logger
import cv2
logger = get_logger()

from backend.progress_bar import AbstractProgressBar


ALIGNMENT_OFFSET = 0.01


def to_gray(img_data):
    return np.dot(img_data[..., :3], [0.2989, 0.5870, 0.1140])

def stretch_image(img_data):
    return Stretch().stretch(img_data)

def debayer(img_data):
    return np.array(cv2.cvtColor(img_data, cv2.COLOR_BayerBG2BGR), dtype='float32')

@arg_logger
def get_master_dark( folder, to_debayer=False, darks_number_limit=10):
    logger.log.info("Making master dark")
    files = [
        os.path.join(folder, item) for item in os.listdir(folder) if
        item.lower().endswith(".fit") or item.lower().endswith(".fits")]
    if not files:
        logger.log.warn("There are no FITS files located in the dark folder. Continuing without darks.")
        return
    imgs = []
    for file_name in files[:darks_number_limit]:
        img = astropy.io.fits.getdata(file_name)
        if len(img.shape) == 2:
            img.shape = *img.shape, 1
        if img.shape[0] in [1, 3]:
            img = np.swapaxes(img, 0, 2)
        if to_debayer and img.shape[2] == 1:
            img = debayer(img)
        imgs.append(img)
    imgs = np.array(imgs)
    master_dark = np.average(imgs, axis=0)
    return master_dark


def __process_img_data(img_data, non_linear):
    # sometimes image is returned in channels first format. Converting to channels last in this case
    if img_data.shape[0] in [1, 3]:
        img_data = np.swapaxes(img_data, 0, 2)
    # converting to grascale
    if img_data.shape[-1] == 3:
        img_data = to_gray(img_data)
    # convert to 2 dims array
    img_data.shape = *img_data.shape[:2],

    # rotate image to have bigger width than height
    if img_data.shape[0] > img_data.shape[1]:
        img_data = np.swapaxes(img_data, 0, 1)
    # Stretch image if it's in linear state

    img_data = img_data.astype('float32')
    img_data = np.ascontiguousarray(img_data)
    return img_data


def __get_datetime_from_str(timestamp):
    datetime_reg = re.compile("(\d{4}-\d{2}-\d{2}).*(\d{2}:\d{2}:\d{2})")
    match = datetime_reg.match(timestamp)
    date_part = match.group(1)
    time_part = match.group(2)
    year, month, day = date_part.split("-")
    year, month, day = int(year), int(month), int(day)
    hour, minute, second = time_part.split(":")
    hour, minute, second = int(hour), int(minute), int(second)
    timestamp = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)
    return timestamp


def load_fits(file_path, non_linear=False, load_image=True, to_debayer=False, master_dark=None):
    with astropy.io.fits.open(file_path) as hdul:
        header = hdul[0].header
        exposure = float(header['EXPTIME'])
        timestamp = __get_datetime_from_str(header['DATE-OBS'])
        if load_image:
            img_data = np.array(hdul[0].data)
            if len(img_data.shape) == 2:
                img_data.shape = *img_data.shape, 1
            if img_data.shape[0] in [1, 3]:
                img_data = np.swapaxes(img_data, 0, 2)
            if to_debayer and img_data.shape[2] == 1:
                img_data = np.array(debayer(img_data), dtype='float32')
            if master_dark is not None:
                img_data = img_data - master_dark

            if img_data.shape[2] == 3:
                img_data = to_gray(img_data)
            img_data = __process_img_data(img_data, non_linear)
        else:
            img_data = None
    return img_data, timestamp, exposure


def load_xisf(file_path, non_linear=False, load_image=True, to_debayer=False, master_dark=None):
    _ = to_debayer
    _ = master_dark
    xisf = XISF(file_path)
    img_meta = xisf.get_images_metadata()[0]
    timestamp = __get_datetime_from_str(img_meta["FITSKeywords"]["DATE-OBS"][0]['value'])
    exposure = float(img_meta["FITSKeywords"]["EXPTIME"][0]['value'])
    if load_image:
        img_data = xisf.read_image(0)
        img_data = np.array(img_data)
        img_data = __process_img_data(img_data, non_linear)
    else:
        img_data = None
    return img_data, timestamp, exposure


def load_worker(load_fps, num_list, imgs_shape, shared_mem_names, load_func, non_linear, progress_queue: Queue, error_queue: Queue,
                reference_image, to_align, to_skip_bad, to_debayer, master_dark):
    try:
        img_buf_name, y_buf_name, x_buf_name = shared_mem_names
        imgs_buf = SharedMemory(name=img_buf_name, create=False)
        loaded_images = np.ndarray(imgs_shape, dtype='float32', buffer=imgs_buf.buf)
        y_boar_buf = SharedMemory(name=y_buf_name, create=False)
        y_boar = np.ndarray((imgs_shape[0], 2), dtype='uint16', buffer=y_boar_buf.buf)
        x_boar_buf = SharedMemory(name=x_buf_name, create=False)
        x_boar = np.ndarray((imgs_shape[0], 2), dtype='uint16', buffer=x_boar_buf.buf)
        for num, fp in zip(num_list, load_fps):
            if not error_queue.empty():
                return
            rejected = False
            img_data, _, _ = load_func(fp, non_linear, load_image=True, to_debayer=to_debayer, master_dark=master_dark)
            if to_align:
                try:
                    img_data, _ = aa.register(img_data, reference_image, fill_value=0)
                except (aa.MaxIterError, ValueError):
                    if to_skip_bad:
                        rejected = True
                    else:
                        raise Exception("Unable to make star alignment. Try to delete bad images or use --skip_bad key")
            if not rejected:
                if not non_linear:
                    img_data = stretch_image(img_data)
                img_data += ALIGNMENT_OFFSET
                y_boarders, x_boarders = SourceData.crop_raw(img_data, to_do=False)
                y_boarders, x_boarders = SourceData.crop_fine(
                    img_data, y_pre_crop_boarders=y_boarders, x_pre_crop_boarders=x_boarders, to_do=False)
                img_data -= ALIGNMENT_OFFSET


                y_boar[num] = np.array(y_boarders, dtype="uint16")
                x_boar[num] = np.array(x_boarders, dtype="uint16")
                loaded_images[num] = img_data
            progress_queue.put((num, rejected))
        imgs_buf.close()
    except:
        error_trace = traceback.format_exc()
        progress_queue.put(error_trace)
        error_queue.put(error_trace)
        return

def get_file_paths(folder):
    return [os.path.join(folder, item) for item in os.listdir(folder) if item.lower().endswith(".xisf") or item.lower().endswith(".fit") or item.lower().endswith(".fits")]


class SourceData:
    ZERO_TOLERANCE = 100
    BOARDER_OFFSET = 10
    X_SPLITS = 1
    Y_SPLITS = 1

    def __init__(self, file_list, non_linear=False, to_align=True, to_skip_bad=False, num_from_session=None, to_debayer=False):
        self.file_list = file_list
        self.to_align = to_align
        self.to_skip_bad = to_skip_bad
        self.non_linear = non_linear
        self.imgs_shm = None
        self.y_boarders_shm = None
        self.x_boarders_shm = None
        self.raw_dataset = None
        self.img_shape = None
        self.exclusion_boxes = None
        self.timestamped_file_list = None
        self.images = None
        self.num_from_session = num_from_session
        self.to_debayer = to_debayer

    def __del__(self):
        if isinstance(self.imgs_shm, SharedMemory):
            self.imgs_shm.unlink()
        if isinstance(self.y_boarders_shm, SharedMemory):
            self.y_boarders_shm.unlink()
        if isinstance(self.x_boarders_shm, SharedMemory):
            self.x_boarders_shm.unlink()

    @property
    def timestamps(self):
        return tuple(item[1] for item in self.timestamped_file_list)

    @property
    def exposures(self):
        return tuple(item[2] for item in self.timestamped_file_list)

    # @classmethod
    # def stretch_image(cls, img_data):
    #     return Stretch().stretch(img_data) + ALIGNMENT_OFFSET

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
        y_bottom, x_right = img_data.shape[:2]
        for num, line in enumerate(img_data):
            if np.any(line):
                y_top = num
                break
        for num, line in enumerate(img_data[::-1]):
            if np.any(line):
                y_bottom -= num
                break

        for num, line in enumerate(img_data.T):
            if np.any(line):
                x_left = num
                break

        for num, line in enumerate(img_data.T[::-1]):
            if np.any(line):
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

        def get_num_of_corner_zeros(line):
            zeros_num = 0
            for item in line:
                if item == 0:
                    zeros_num += 1
                else:
                    break
            for item in line[::-1]:
                if item == 0:
                    zeros_num += 1
                else:
                    break
            return zeros_num

        y_top_zeros = get_num_of_corner_zeros(pre_cropped[0])
        y_bottom_zeros = get_num_of_corner_zeros(pre_cropped[-1])
        x_left_zeros = get_num_of_corner_zeros(pre_cropped.T[0])
        x_right_zeros = get_num_of_corner_zeros(pre_cropped.T[-1])
        zeros = y_top_zeros, y_bottom_zeros, x_left_zeros, x_right_zeros
        trim_args = (1, False), (-1, False), (1, True), (-1, True)
        args_order = (item[1] for item in sorted(zip(zeros, trim_args), key=lambda x: x[0], reverse=True))

        def _fine_crop_border(img_data_tmp, direction, transpon=True):
            if transpon:
                img_data_tmp = img_data_tmp.T
            x = 0
            for num, line in enumerate(img_data_tmp[::direction]):
                if get_num_of_corner_zeros(line) <= get_num_of_corner_zeros(
                        img_data_tmp[::direction][num + 1]) and get_num_of_corner_zeros(line) < cls.ZERO_TOLERANCE:
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

    def prepare_timestamps(self, timestamps):
        normalized_timestamps, diff_timestamps = self.normalize_timestamps(timestamps)
        return normalized_timestamps, diff_timestamps

    def load_headers_and_sort(self):
        timestamped_file_list = []
        for fp in self.file_list:
            if fp.lower().endswith(".xisf"):
                load_func = load_xisf
            elif fp.lower().endswith(".fits") or fp.lower().endswith(".fit"):
                load_func = load_fits
            else:
                raise ValueError("xisf or fit/fits files are allowed")

            _, timestamp, exposure = load_func(fp, self.non_linear, load_image=False)
            timestamped_file_list.append((fp, timestamp, exposure))
        timestamped_file_list.sort(key=lambda x: x[1])

        exposures = [item[2] for item in timestamped_file_list]
        timestamps = [item[1] for item in timestamped_file_list]
        file_paths = [item[0] for item in timestamped_file_list]
        if self.num_from_session is not None:
            new_file_paths = []
            new_timestamps = []
            new_exposures = []
            add_number = 0
            for num in range(len(timestamps)):
                # new session if timestamp diss is more than 10 hours
                if timestamps[num] - timestamps[num-1 if num-1 >= 0 else 0] > datetime.timedelta(hours=10):
                    add_number = 0
                if add_number < self.num_from_session:
                    new_file_paths.append(file_paths[num])
                    new_timestamps.append(timestamps[num])
                    new_exposures.append(exposures[num])
                    add_number += 1
            file_paths = new_file_paths
            timestamps = new_timestamps
            exposures = new_exposures
        self.timestamped_file_list = list(zip(file_paths, timestamps, exposures))

    @arg_logger
    def load_images(self, to_debayer: bool = False, dark_folder=None, progress_bar: Optional[AbstractProgressBar] = None, frame: Optional[wx.Frame] = None):
        to_align = self.to_align
        non_linear = self.non_linear
        file_paths = [item[0] for item in self.timestamped_file_list]
        if progress_bar:
            progress_bar.set_total(len(file_paths))

        if file_paths[0].lower().endswith(".xisf"):
            load_func = load_xisf
        elif file_paths[0].lower().endswith(".fits") or file_paths[0].lower().endswith(".fit"):
            load_func = load_fits
        else:
            raise ValueError("xisf or fit/fits files are allowed")
        img, _, _ = load_func(file_paths[0], non_linear, True, to_debayer=to_debayer)
        logger.log.debug(f"Initial image shape: {img.shape}")
        mem_size = 1
        for n in (len(file_paths), *img.shape):
            mem_size = mem_size * n
        mem_size = mem_size * 4
        logger.log.debug(f"Allocating memory size: {mem_size // (1024 * 1024)} Mb")
        self.imgs_shm = SharedMemory(size=mem_size, create=True)
        logger.log.debug(f"Allocated memory size: {mem_size // (1024 * 1024)} Mb")
        images_shape = (len(file_paths), *img.shape[:2])
        imgs = np.ndarray(images_shape, dtype='float32', buffer=self.imgs_shm.buf)

        self.y_boarders_shm = SharedMemory(size=len(file_paths) * 2 * 2, create=True)
        y_boaredrs = np.ndarray((len(file_paths), 2), dtype='uint16', buffer=self.y_boarders_shm.buf)
        self.x_boarders_shm = SharedMemory(size=len(file_paths) * 2 * 2, create=True)
        x_boaredrs = np.ndarray((len(file_paths), 2), dtype='uint16', buffer=self.x_boarders_shm.buf)


        master_dark = None
        if dark_folder is not None and os.path.exists:
            master_dark = get_master_dark(dark_folder, to_debayer=to_debayer)

        worker_num = min((os.cpu_count(), 4))
        processes = []
        # progress_bar = tqdm.tqdm(total=len(file_paths))
        progress_queue = Queue()
        error_queue = Queue()
        for num in range(worker_num):
            logger.log.debug(f"Running loading process {num}")
            processes.append(
                Process(target=load_worker, args=(
                    file_paths[num::worker_num],
                    list(range(len(file_paths)))[num::worker_num],
                    images_shape,
                    (self.imgs_shm.name, self.y_boarders_shm.name, self.x_boarders_shm.name),
                    load_func,
                    non_linear,
                    progress_queue,
                    error_queue,
                    img,
                    to_align,
                    True,
                    to_debayer,
                    master_dark
                ))
            )
        for proc in processes:
            proc.start()

        rejected_list = []
        for _ in range(len(file_paths)):
            res = progress_queue.get()
            if isinstance(res, str):
                logger.log.error(f"Unexpected error happened during loading images:\n{res}")
                for proc in processes:
                    proc.join()
                sys.exit(-1)
            num, rejected = res
            if rejected:
                rejected_list.append(num)
                logger.log.info(f"Rejected file: {self.timestamped_file_list[num][0]}")

            if progress_bar:
                progress_bar.update()

        for proc in processes:
            proc.join()

        if len(rejected_list):
            imgs = np.delete(imgs, rejected_list, axis=0)
            x_boaredrs = np.delete(x_boaredrs, rejected_list, axis=0)
            y_boaredrs = np.delete(y_boaredrs, rejected_list, axis=0)
            for item in rejected_list:
                self.timestamped_file_list[item] = None

            self.timestamped_file_list = [item for item in self.timestamped_file_list if item is not None]

        y_boarders = int(np.max(y_boaredrs[:, 0])), int(np.min(y_boaredrs[:, 1]))
        x_boarders = int(np.max(x_boaredrs[:, 0])), int(np.min(x_boaredrs[:, 1]))
        x_left, x_right = x_boarders
        y_top, y_bottom = y_boarders
        imgs = imgs[:, y_top: y_bottom, x_left: x_right]
        # imgs = self.chop_imgs(imgs)
        self.images = imgs
        self.img_shape = self.images[0].shape
        if frame:
            wx.CallAfter(frame.on_load_finished)

    @classmethod
    def normalize_timestamps(cls, timestamps):
        timestamps = [(item - min(timestamps)).total_seconds() for item in timestamps]
        new_timestamps = []
        first_ts = timestamps[0]
        for item in timestamps:
            if item - first_ts > 14 * 60 * 60:
                first_ts = item
            new_timestamps.append(item - first_ts)
        normalized_timestamps = np.array([item / max(new_timestamps) for item in new_timestamps])

        diff_timestamps = np.array(
            [timestamps[i] - timestamps[i - 1 if i - 1 >= 0 else 0] for i in range(len(timestamps))])
        diff_timestamps = (
            diff_timestamps - np.min(diff_timestamps)) / (np.max(diff_timestamps) - np.min(diff_timestamps))
        return np.array([normalized_timestamps, diff_timestamps])

    def __get_exclusion_boxes_paths(self):
        folders = {os.path.dirname(path) for path in self.file_list}
        exclusion_boxes_files = []
        for folder in folders:
            if "annotations.xml" in os.listdir(folder):
                exclusion_boxes_files.append(os.path.join(folder, "annotations.xml"))
        return exclusion_boxes_files

    def load_exclusion_boxes(self):
        # Reading data from the xml file
        img_shape = self.images[0].shape[:2]
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
                y_shape, x_shape = img_shape
                y_mult = y_shape / height
                x_mult = x_shape / width

            for tag in bs_data.find_all('box', {'label': 'Asteroid'}):
                xtl = round(float(tag.get("xtl")) * x_mult)
                ytl = round(float(tag.get("ytl")) * y_mult)
                xbr = round(float(tag.get("xbr")) * x_mult)
                ybr = round(float(tag.get("ybr")) * y_mult)
                boxes.append((xtl, ytl, xbr, ybr))

            all_boxes.extend(boxes)
        self.exclusion_boxes = np.array(all_boxes)

    def chop_imgs(self, y_splits, x_splits):
        shape = self.images[0].shape
        y_shape, x_shape = shape[:2]
        y_split_size = (y_shape - 2 * self.BOARDER_OFFSET) // y_splits
        x_split_size = (x_shape - 2 * self.BOARDER_OFFSET) // x_splits
        self.images = self.images[
           :,
           :y_split_size * y_splits + 2 * self.BOARDER_OFFSET,
           :x_split_size * x_splits + 2 * self.BOARDER_OFFSET,
        ]

    @arg_logger
    def gen_splits(self, y_splits, x_splits, to_align=True, use_img_mask=None, to_skip_bad=True):
        shape = self.images[0].shape
        y_shape, x_shape = shape[:2]
        y_split_size = (y_shape - 2 * self.BOARDER_OFFSET) // y_splits
        x_split_size = (x_shape - 2 * self.BOARDER_OFFSET) // x_splits

        for y_num in range(y_splits):
            for x_num in range(x_splits):
                new_use_img_mask = copy.copy(use_img_mask)
                y_offset = self.BOARDER_OFFSET + y_num * y_split_size
                x_offset = self.BOARDER_OFFSET + x_num * x_split_size
                not_aligned = self.images[
                              :,
                              y_offset - self.BOARDER_OFFSET: y_offset + y_split_size + self.BOARDER_OFFSET,
                              x_offset - self.BOARDER_OFFSET: x_offset + x_split_size + self.BOARDER_OFFSET]
                if to_align:
                    aligned = np.ndarray((len(self.images), y_split_size, x_split_size), dtype='float32')
                    aligned[0] = not_aligned[0, self.BOARDER_OFFSET: -self.BOARDER_OFFSET, self.BOARDER_OFFSET: -self.BOARDER_OFFSET]
                    bad = []
                    for num in range(1, len(not_aligned)):
                        try:
                            new_img, _ = aa.register(not_aligned[num], aligned[0], fill_value=0)
                            aligned[num] = new_img
                        except:
                            if to_skip_bad:
                                bad.append(num)
                                new_use_img_mask[num] = False

                            else:
                                raise Exception(
                                    "Unable to make star alignment. Try to delete bad images or use --skip_bad key")
                    # if bad:
                    #     aligned = np.delete(aligned, bad, axis=0)
                else:
                    aligned = not_aligned[:, self.BOARDER_OFFSET: -self.BOARDER_OFFSET, self.BOARDER_OFFSET: -self.BOARDER_OFFSET]

                logger.log.debug("SPLIT GENERATED")
                yield aligned, y_offset, x_offset, new_use_img_mask

    def get_max_image(self, img_mask=None):
        if img_mask is None:
            img_mask = [True] * len(self.images)
        return np.amax(self.images[img_mask], axis=0)

    @classmethod
    def get_shrinked_img_series(cls, images, size, y, x, img_mask=None):
        if img_mask is None:
            img_mask = [True] * len(images)
        shrinked = np.copy(images[img_mask, y:y+size, x:x+size])
        return shrinked

    @classmethod
    def prepare_images(cls, imgs):
        imgs = np.array(
            [np.amax(np.array([imgs[num] - imgs[0], imgs[num] - imgs[-1]]), axis=0) for num in range(len(imgs))])
        imgs[imgs < 0] = 0
        imgs = (imgs - np.min(imgs)) / (np.max(imgs) - np.min(imgs))
        imgs.shape = (*imgs.shape, 1)
        return imgs

    # @staticmethod
    def adjust_series_to_min_len(self, imgs, timestamps, min_len=8):
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


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
from PIL import Image

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


ALIGNMENT_OFFSET = 0.5


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
        img /= 256 * 256 - 1
        imgs.append(img)
    imgs = np.array(imgs)
    master_dark = np.average(imgs, axis=0)
    return master_dark

@arg_logger
def get_master_flat(flat_folder, dark_flat_folder, to_debayer=False, flats_number_limit=10):
    logger.log.info("Making master flat")
    flat_dark = get_master_dark(dark_flat_folder, to_debayer)
    files = [
        os.path.join(flat_folder, item) for item in os.listdir(flat_folder) if
            item.lower().endswith(".fit") or item.lower().endswith(".fits")]
    if not files:
        logger.log.warn("There are no FITS files located in the flat folder. Continuing without darks.")
        return
    imgs = []
    for file_name in files[:flats_number_limit]:
        img = astropy.io.fits.getdata(file_name)
        if len(img.shape) == 2:
            img.shape = *img.shape, 1
        if img.shape[0] in [1, 3]:
            img = np.swapaxes(img, 0, 2)
        if to_debayer and img.shape[2] == 1:
            img = debayer(img)
        img /= 256 * 256 - 1
        imgs.append(img - flat_dark)
    imgs = np.array(imgs)
    master_flat = np.average(imgs, axis=0)
    return master_flat


def process_img_data(img_data, non_linear):
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


def load_fits(file_path, load_image=True, to_debayer=False):
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
            # Normalize
            img_data /= 256 * 256 - 1
        else:
            img_data = None
    return img_data, timestamp, exposure


def load_xisf(file_path, load_image=True, to_debayer=False):
    _ = to_debayer
    xisf = XISF(file_path)
    img_meta = xisf.get_images_metadata()[0]
    timestamp = __get_datetime_from_str(img_meta["FITSKeywords"]["DATE-OBS"][0]['value'])
    exposure = float(img_meta["FITSKeywords"]["EXPTIME"][0]['value'])
    if load_image:
        img_data = xisf.read_image(0)
        img_data = np.array(img_data)

    else:
        img_data = None
    return img_data, timestamp, exposure


def load_worker(load_fps, num_list, load_func, non_linear, progress_queue: Queue, error_queue: Queue,
                reference_image, to_align, to_skip_bad, to_debayer, master_dark, master_flat):
    reference_image = reference_image * reference_image
    # reference_image[reference_image < 0] = 0
    try:
        for num, fp in zip(num_list, load_fps):
            if not error_queue.empty():
                return
            rejected = False
            img_data, _, _ = load_func(fp, load_image=True, to_debayer=to_debayer)
            if master_dark is not None:
                img_data -= master_dark
            if master_flat is not None:
                img_data /= master_flat
            img_data = process_img_data(img_data, non_linear)
            # if not non_linear:
            #     img_data = stretch_image(img_data)
            # if to_align:
            #     img_data += ALIGNMENT_OFFSET
            if to_align:
                try:
                    img_data, _ = aa.register(img_data, reference_image, 0, max_control_points=50, min_area=5)
                except:
                    if to_skip_bad:
                        rejected = True
                    else:
                        raise Exception("Unable to make star alignment. Try to delete bad images or use --skip_bad key")
            if not rejected:
                y_boarders, x_boarders = SourceData.crop_raw(img_data, to_do=False)

                y_boarders, x_boarders = SourceData.crop_fine(
                    img_data, y_pre_crop_boarders=y_boarders, x_pre_crop_boarders=x_boarders, to_do=False)
                # if to_align:
                #     img_data -= ALIGNMENT_OFFSET
                progress_queue.put((num, rejected, img_data, np.array(y_boarders, dtype="uint16"), np.array(x_boarders, dtype="uint16")))
            else:
                progress_queue.put((num, rejected, None, np.array((0, 0), dtype="uint16"), np.array((0, 0), dtype="uint16")))
    except:
        error_trace = traceback.format_exc()
        progress_queue.put(error_trace)
        error_queue.put(error_trace)
        return


def get_file_paths(folder):
    return [os.path.join(folder, item) for item in os.listdir(folder) if item.lower().endswith(".xisf") or item.lower().endswith(".fit") or item.lower().endswith(".fits")]


class SourceData:
    ZERO_TOLERANCE = 100
    BOARDER_OFFSET = 20
    X_SPLITS = 1
    Y_SPLITS = 1

    def __init__(self, file_list, non_linear=False, to_align=True, to_skip_bad=False, num_from_session=None, to_debayer=False, dark_folder=None, flat_folder=None, dark_flat_folder=None):
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
        self.dark_folder = dark_folder
        self.flat_folder = flat_folder
        self.dark_flat_folder = dark_flat_folder

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

            _, timestamp, exposure = load_func(fp, load_image=False)
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
    def load_images(self, progress_bar: Optional[AbstractProgressBar] = None, frame: Optional[wx.Frame] = None):
        to_align = self.to_align
        non_linear = self.non_linear
        to_debayer = self.to_debayer
        file_paths = [item[0] for item in self.timestamped_file_list]
        if file_paths[0].lower().endswith(".xisf"):
            load_func = load_xisf
        elif file_paths[0].lower().endswith(".fits") or file_paths[0].lower().endswith(".fit"):
            load_func = load_fits
        else:
            raise ValueError("xisf or fit/fits files are allowed")
        template_img, _, _ = load_func(file_paths[0], True, to_debayer=to_debayer)
        logger.log.debug(f"Initial image shape: {template_img.shape}")
        mem_size = 1
        for n in (len(file_paths), *template_img.shape):
            mem_size = mem_size * n
        mem_size = mem_size * 4
        logger.log.debug(f"Allocating memory size: {mem_size // (1024 * 1024)} Mb")
        logger.log.debug(f"Allocated memory size: {mem_size // (1024 * 1024)} Mb")
        images_shape = (len(file_paths), *template_img.shape[:2])
        imgs = np.ndarray(images_shape, dtype='float32')

        y_boaredrs = np.ndarray((len(file_paths), 2), dtype='uint16')
        x_boaredrs = np.ndarray((len(file_paths), 2), dtype='uint16')
        master_dark = None
        master_flat = None
        if self.dark_folder is not None and os.path.exists(self.dark_folder):
            master_dark = get_master_dark(self.dark_folder, to_debayer=to_debayer)
        if self.flat_folder is not None and self.dark_flat_folder is not None and os.path.exists(self.flat_folder) and \
                os.path.exists(self.dark_flat_folder):
            master_flat = get_master_flat(self.flat_folder, self.dark_flat_folder, to_debayer=to_debayer)
        if master_dark is not None:
            template_img -= master_dark
        if master_flat is not None:
            template_img /= master_flat
        template_img = process_img_data(template_img, non_linear)
        # if not non_linear:
        #     template_img = stretch_image(template_img)
        worker_num = min((os.cpu_count(), 4))
        processes = []
        progress_queue = Queue()
        error_queue = Queue()
        for num in range(worker_num):
            logger.log.debug(f"Running loading process {num}")
            processes.append(
                Process(target=load_worker, args=(
                    file_paths[num::worker_num],
                    list(range(len(file_paths)))[num::worker_num],
                    load_func,
                    non_linear,
                    progress_queue,
                    error_queue,
                    template_img,
                    to_align,
                    True,
                    to_debayer,
                    master_dark,
                    master_flat
                ))
            )
        for proc in processes:
            proc.start()

        if progress_bar:
            progress_bar.set_total(len(file_paths))
        rejected_list = []
        for _ in range(len(file_paths)):
            res = progress_queue.get()
            if isinstance(res, str):
                logger.log.error(f"Unexpected error happened during loading images:\n{res}")
                for proc in processes:
                    proc.join()
                sys.exit(-1)
            num, rejected, loaded_img, loaded_y_borders, loaded_x_borders = res
            imgs[num] = loaded_img
            y_boaredrs[num] = loaded_y_borders
            x_boaredrs[num] = loaded_x_borders
            if rejected:
                rejected_list.append(num)
                logger.log.warning(f"Rejected file: {self.timestamped_file_list[num][0]}")
            else:
                logger.log.debug(f"Loaded file: {self.timestamped_file_list[num][0]}")

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
        if any(ts != 0 for ts in new_timestamps):
            normalized_timestamps = np.array([item / max(new_timestamps) for item in new_timestamps])
        else:
            normalized_timestamps = new_timestamps
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
                xtl = round(float(tag.get("xtl")) * x_mult) - 100
                ytl = round(float(tag.get("ytl")) * y_mult) - 100
                xbr = round(float(tag.get("xbr")) * x_mult) + 100
                ybr = round(float(tag.get("ybr")) * y_mult) + 100
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
    def gen_splits(self, y_splits, x_splits, to_align=True, use_img_mask=None, to_skip_bad=True, output_folder=None):
        logger.log.info(f"Splitting images by {y_splits}x{x_splits}={y_splits*x_splits} parts")
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
                    aligned = np.ndarray((not_aligned.shape[0], not_aligned.shape[1] - 2 * self.BOARDER_OFFSET, not_aligned.shape[2] - 2 * self.BOARDER_OFFSET,), dtype='float32')
                    reference = np.copy(not_aligned[0])

                    aligned[0] = reference[self.BOARDER_OFFSET: -self.BOARDER_OFFSET, self.BOARDER_OFFSET: -self.BOARDER_OFFSET]
                    if not self.non_linear:
                        aligned[0] = stretch_image(aligned[0])
                    # aligned[0] = reference[self.BOARDER_OFFSET: -self.BOARDER_OFFSET, self.BOARDER_OFFSET: -self.BOARDER_OFFSET]
                    bad = []
                    for num in range(1, len(not_aligned)):
                        try:
                            # image_to_align = stretch_image(not_aligned[num])
                            image_to_align = not_aligned[num]
                            # new_img, _ = aa.register(not_aligned[num], reference, fill_value=0, max_control_points=50, min_area=3)
                            transform, *_ = aa.find_transform(not_aligned[num], reference, max_control_points=50, min_area=3)
                            new_img, _ = aa.apply_transform(transform, image_to_align, reference, fill_value=0)
                            new_img = new_img[self.BOARDER_OFFSET: -self.BOARDER_OFFSET, self.BOARDER_OFFSET: -self.BOARDER_OFFSET]
                            if not self.non_linear:
                                new_img = stretch_image(new_img)
                            aligned[num] = new_img
                            # aligned[num] = new_img[self.BOARDER_OFFSET: -self.BOARDER_OFFSET, self.BOARDER_OFFSET: -self.BOARDER_OFFSET]
                        except:
                            if to_skip_bad:
                                logger.log.info(f"Frame {num} in split {y_num} {x_num} was not aligned")
                                bad.append(num)
                                new_use_img_mask[num] = False
                            else:
                                raise Exception(
                                    "Unable to make star alignment. Try to delete bad images or use --skip_bad key")
                    # aligned = aligned[:, self.BOARDER_OFFSET: -self.BOARDER_OFFSET, self.BOARDER_OFFSET: -self.BOARDER_OFFSET]
                else:
                    aligned = not_aligned[:, self.BOARDER_OFFSET: -self.BOARDER_OFFSET, self.BOARDER_OFFSET: -self.BOARDER_OFFSET]
                    if not self.non_linear:
                        for num in range(len(aligned)):
                            aligned[num] = stretch_image(aligned[num])

                # if output_folder:
                #     frames = np.copy(aligned)
                #     frames = frames * 255
                #     new_frames = [Image.fromarray(frame).convert('L').convert('P') for frame in frames]
                #     new_frames[0].save(
                #         os.path.join(output_folder, f"{y_offset}_{x_offset}.gif"),
                #         save_all=True,
                #         append_images=new_frames[1:],
                #         duration=200,
                #         loop=0)

                yield aligned, y_offset, x_offset, new_use_img_mask

    def get_max_image(self, img_mask=None):
        if img_mask is None:
            img_mask = [True] * len(self.images)
        max_image = np.amax(self.images[img_mask], axis=0)
        if not self.non_linear:
            max_image = stretch_image(max_image)
        return max_image

    @classmethod
    def get_shrinked_img_series(cls, images, size, y, x, img_mask=None):
        if img_mask is None:
            img_mask = [True] * len(images)
        shrinked = np.copy(images[img_mask, y:y+size, x:x+size])
        return shrinked

    @classmethod
    def prepare_images(cls, imgs):
        imgs = np.array([stretch_image(item) for item in imgs])
        # imgs = np.array(
        #     [np.amax(np.array([imgs[num] - imgs[0], imgs[num] - imgs[-1]]), axis=0) for num in range(len(imgs))])
        imgs[imgs < 0] = 0
        if np.max(imgs) - np.min(imgs) != 0:
            imgs = (imgs - np.min(imgs)) / (np.max(imgs) - np.min(imgs))
        imgs.shape = (*imgs.shape, 1)
        return imgs

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
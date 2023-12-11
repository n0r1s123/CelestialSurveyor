import datetime
import os
import queue
import re

import astropy.io.fits
import time

import numpy as np
import tqdm

from auto_stretch.stretch import Stretch
from xisf import XISF
from PIL import Image
from bs4 import BeautifulSoup
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory


def __process_img_data(img_data, non_linear):
    # sometimes image is returned in channels first format. Converting to channels last in this case
    if img_data.shape[0] in [1, 3]:
        img_data = np.swapaxes(img_data, 0, 2)
    # converting to grascale
    if img_data.shape[-1] == 3:
        img_data = Dataset.to_gray(img_data)
    # convert to 2 dims array
    img_data.shape = *img_data.shape[:2],

    # rotate image to have bigger width than height
    if img_data.shape[0] > img_data.shape[1]:
        img_data = np.swapaxes(img_data, 0, 1)
    # Stretch image if it's in linear state
    if not non_linear:
        img_data = Dataset.stretch_image(img_data)
    img_data = img_data.astype('float32')
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


def load_fits(file_path, non_linear=False, load_image=True):
    with astropy.io.fits.open(file_path) as hdul:
        header = hdul[0].header
        exposure = float(header['EXPTIME'])
        timestamp = __get_datetime_from_str(header['DATE-OBS'])
        if load_image:
            img_data = hdul[0].data
            img_data = __process_img_data(img_data, non_linear)
        else:
            img_data = None
    return img_data, timestamp, exposure



def load_xisf(file_path, non_linear=False, load_image=True):
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


def load_worker(load_fps, num_list, imgs_shape, shared_mem_names, load_func, non_linear, progress_queue: Queue):
    img_buf_name, y_buf_name, x_buf_name = shared_mem_names
    imgs_buf = SharedMemory(name=img_buf_name, create=False)
    loaded_images = np.ndarray(imgs_shape, dtype='float32', buffer=imgs_buf.buf)
    y_boar_buf = SharedMemory(name=y_buf_name, create=False)
    y_boar = np.ndarray((imgs_shape[0], 2), dtype='uint16', buffer=y_boar_buf.buf)
    x_boar_buf = SharedMemory(name=x_buf_name, create=False)
    x_boar = np.ndarray((imgs_shape[0], 2), dtype='uint16', buffer=x_boar_buf.buf)

    # print(num_list, load_fps)
    for num, fp in zip(num_list, load_fps):

        img_data, _, _ = load_func(fp, non_linear, load_image=True)
        loaded_images[num] = img_data
        y_boarders, x_boarders = Dataset.crop_raw(img_data, to_do=False)
        y_boarders, x_boarders = Dataset.crop_fine(
            img_data, y_pre_crop_boarders=y_boarders, x_pre_crop_boarders=x_boarders, to_do=False)
        y_boar[num] = np.array(y_boarders, dtype="uint16")
        x_boar[num] = np.array(x_boarders, dtype="uint16")
        progress_queue.put(num)
    imgs_buf.close()
        
        # print(y_boar)
        # boarders = np.append(boarders, np.array([*y_boarders, *x_boarders]))

class SourceData:
    def __init__(self, folders=None, samples_folder=None, non_linear=False):

        self.imgs_shm = None
        self.y_boarders_shm = None
        self.x_boarders_shm = None
        self.raw_dataset, self.exposures, self.timestamps, self.img_shape, self.exclusion_boxes = self.__load_raw_dataset(folders, non_linear)
        normalized_timestamps = [[(item - min(timestamps)).total_seconds() for item in timestamps] for timestamps in self.timestamps]
        self.normalized_timestamps = [np.array(
            [item / max(timestamps) for item in timestamps]) for timestamps in normalized_timestamps]
        diff_timestamps = [np.array([(timestamps[i] - timestamps[i-1]).total_seconds() for i in range(len(timestamps))]) for timestamps in self.timestamps]
        self.diff_timestamps = [(diffs - np.min(diffs)) / (np.max(diffs) - np.min(diffs)) for diffs in diff_timestamps]
        # for diffs in diff_timestamps:
        #     (diffs - np.min(diffs)) / (np.max(diffs) - np.min(diffs))
        if samples_folder is not None and os.path.exists(samples_folder):
            self.object_samples = self.__load_samples(samples_folder)
        else:
            self.object_samples = []


    def __load_raw_dataset(self, folders, non_linear=False):
        raw_dataset = [self.crop_folder_on_the_fly(folder, non_linear) for folder in folders]
        all_timestamps = [item[1] for item in raw_dataset]
        all_exposures = [item[2] for item in raw_dataset]
        raw_dataset = [item[0] for item in raw_dataset]
        all_exclusion_boxes = []
        img_shapes = []
        for num1, folder in enumerate(folders):
            img_shape = raw_dataset[num1].shape[1:]
            exclusion_boxes = self.__load_exclusion_boxes(folder, img_shape)
            all_exclusion_boxes.append(exclusion_boxes)
            img_shapes.append(img_shape)

        print("Raw image data loaded:")
        print(f"SHAPE: {[item.shape for item in raw_dataset]}")
        print(f"Used RAM: {sum([item.itemsize * item.size for item in raw_dataset]) // (1024 * 1024)} Mb")

        return raw_dataset, all_exposures, all_timestamps, img_shapes, all_exclusion_boxes

    @classmethod
    def __load_samples(cls, samples_folder):
        file_list = [os.path.join(samples_folder, item) for item in os.listdir(samples_folder) if ".tif" in item]
        samples = np.array([np.array(Image.open(item)) for item in file_list])
        return samples

    @classmethod
    def __load_exclusion_boxes(cls, folder, img_shape):
        # Reading data from the xml file
        fp = os.path.join(folder, 'annotations.xml')
        if not os.path.exists(fp):
            return

        with open(fp, 'r') as f:
            data = f.read()
        bs_data = BeautifulSoup(data, 'xml')
        boxes = []
        width = float(bs_data.find('image').get("width"))
        height = float(bs_data.find('image').get("height"))
        if img_shape is None:
            x_mult, y_mult = 1, 1
        else:
            y_shape, x_shape = img_shape[:2]
            y_mult = y_shape / height
            x_mult = x_shape / width

        for tag in bs_data.find_all('box', {'label': 'Asteroid'}):
            xtl = round(float(tag.get("xtl")) * x_mult)
            ytl = round(float(tag.get("ytl")) * y_mult)
            xbr = round(float(tag.get("xbr")) * x_mult)
            ybr = round(float(tag.get("ybr")) * y_mult)
            boxes.append((xtl, ytl, xbr, ybr))

        boxes = np.array(boxes)

        return boxes







    def crop_folder_on_the_fly(self, input_folder, non_linear):
        file_list = [item for item in os.listdir(input_folder) if ".xisf" in item.lower() or ".fits" in item.lower()]
        if not file_list:
            raise ValueError(f"There are no files in '{input_folder}' with extensions "
                             f".fits, .FITS, .fit, .FIT, .xisf, .XISF and other combinations of capital-lower letters")
        if all(item.lower().endswith(".xisf") for item in file_list):
            load_func = load_xisf
        elif all(item.lower().endswith(".fits") or item.lower().endswith(".fit") for item in file_list):
            load_func = load_fits
        else:
            raise ValueError("There are mixed XISF and FITS files in the folder.")
        print("Loading image meta and sorting according to timestamps...")
        time.sleep(0.1)

        progress_bar = tqdm.tqdm(total=len(file_list))
        timestamped_file_list = []
        for item in file_list:
            fp = os.path.join(input_folder, item)
            _, timestamp, exposure = load_func(fp, non_linear, load_image=False)
            timestamped_file_list.append((fp, timestamp, exposure))
            progress_bar.update()
        progress_bar.close()

        timestamped_file_list.sort(key=lambda x: x[1])
        file_paths = [item[0] for item in timestamped_file_list]
        timestamps = [item[1] for item in timestamped_file_list]
        exposures = [item[2] for item in timestamped_file_list]
        print("Loading and cropping images...")
        time.sleep(0.1)


        img, _, _ = load_func(file_paths[0], non_linear, True)
        mem_size = int(np.prod((len(file_paths), *img.shape))) * 4
        self.imgs_shm = SharedMemory(size=mem_size, create=True)
        images_shape = (len(file_paths), *img.shape)
        imgs = np.ndarray(images_shape, dtype='float32', buffer=self.imgs_shm.buf)

        self.y_boarders_shm = SharedMemory(size=len(file_paths) * 2 * 2, create=True)
        y_boaredrs = np.ndarray((len(file_paths), 2), dtype='uint16', buffer=self.y_boarders_shm.buf)
        self.x_boarders_shm = SharedMemory(size=len(file_paths) * 2 * 2, create=True)
        x_boaredrs = np.ndarray((len(file_paths), 2), dtype='uint16', buffer=self.x_boarders_shm.buf)
        # worker_num = os.cpu_count()
        worker_num = min((os.cpu_count(), 4))
        processes = []
        progress_bar = tqdm.tqdm(total=len(file_list))
        progress_queue = Queue()
        for num in range(worker_num):
            processes.append(
                Process(target=load_worker, args=(
                    file_paths[num::worker_num],
                    list(range(len(file_paths)))[num::worker_num],
                    images_shape,
                    (self.imgs_shm.name, self.y_boarders_shm.name, self.x_boarders_shm.name),
                    load_func,
                    non_linear, 
                    progress_queue
                ))
            )
        for proc in processes:
            proc.start()

        for _ in range(len(file_paths)):
            progress_queue.get()
            progress_bar.update()

        for proc in processes:
            proc.join()
        progress_bar.close()

        y_boarders = int(np.max(y_boaredrs[:, 0])), int(np.min(y_boaredrs[:, 1]))
        x_boarders = int(np.max(x_boaredrs[:, 0])), int(np.min(x_boaredrs[:, 1]))
        x_left, x_right = x_boarders
        y_top, y_bottom = y_boarders
        imgs = imgs[:, y_top: y_bottom, x_left: x_right]
        return imgs, timestamps, exposures


class Dataset:
    ZERO_TOLERANCE = 100

    def __init__(self, source_data: SourceData):
        self.source_data = source_data

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
                if get_num_of_corner_zeros(line) <= get_num_of_corner_zeros(img_data_tmp[::direction][num + 1]) and get_num_of_corner_zeros(line) < cls.ZERO_TOLERANCE:
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
    def get_max_image(cls, images):
        return np.amax(images, axis=0)

    @classmethod
    def prepare_images(cls, imgs):
        imgs = np.array(
            [np.amax(np.array([imgs[num] - imgs[0], imgs[num] - imgs[-1]]), axis=0) for num in range(len(imgs))])
        imgs[imgs < 0] = 0
        imgs = (imgs - np.min(imgs)) / (np.max(imgs) - np.min(imgs))
        imgs.shape = (*imgs.shape, 1)
        return imgs

    def get_shrinked_img_series(self, size, y, x, dataset=None, dataset_idx=0):
        dataset = self.source_data.raw_dataset[dataset_idx] if dataset is None else dataset
        shrinked = np.copy(dataset[:, y:y+size, x:x+size])
        return shrinked



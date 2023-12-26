import datetime
import os
import re

import astropy.io.fits
import time

import numpy as np
import tqdm

from auto_stretch.stretch import Stretch
from xisf import XISF
from bs4 import BeautifulSoup
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory
import astroalign as aa



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


def load_fits(file_path, non_linear=False, load_image=True):
    with astropy.io.fits.open(file_path) as hdul:
        header = hdul[0].header
        exposure = float(header['EXPTIME'])
        timestamp = __get_datetime_from_str(header['DATE-OBS'])
        if load_image:
            img_data = hdul[0].data
            img_data = np.array(img_data)
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


def load_worker(load_fps, num_list, imgs_shape, shared_mem_names, load_func, non_linear, progress_queue: Queue,
                reference_image, to_align, to_skip_bad):
    img_buf_name, y_buf_name, x_buf_name = shared_mem_names
    imgs_buf = SharedMemory(name=img_buf_name, create=False)
    loaded_images = np.ndarray(imgs_shape, dtype='float32', buffer=imgs_buf.buf)
    y_boar_buf = SharedMemory(name=y_buf_name, create=False)
    y_boar = np.ndarray((imgs_shape[0], 2), dtype='uint16', buffer=y_boar_buf.buf)
    x_boar_buf = SharedMemory(name=x_buf_name, create=False)
    x_boar = np.ndarray((imgs_shape[0], 2), dtype='uint16', buffer=x_boar_buf.buf)

    for num, fp in zip(num_list, load_fps):

        rejected = False
        img_data, _, _ = load_func(fp, non_linear, load_image=True)
        if to_align:
            try:
                img_data, _ = aa.register(img_data, reference_image, fill_value=0)
            except aa.MaxIterError:
                if to_skip_bad:
                    rejected = True
                else:
                    raise Exception("Unable to make star alignment. Try to delete bad images or use --skip_bad key")
        if not rejected:
            loaded_images[num] = img_data
            y_boarders, x_boarders = Dataset.crop_raw(img_data, to_do=False)
            y_boarders, x_boarders = Dataset.crop_fine(
                img_data, y_pre_crop_boarders=y_boarders, x_pre_crop_boarders=x_boarders, to_do=False)
            y_boar[num] = np.array(y_boarders, dtype="uint16")
            x_boar[num] = np.array(x_boarders, dtype="uint16")
        progress_queue.put((num, rejected))
    imgs_buf.close()


class SourceData:
    BOARDER_OFFSET = 10
    X_SPLITS = 3
    Y_SPLITS = 3

    def __init__(self, folder, non_linear=False, num_from_session=0, to_align=True, to_skip_bad=False):
        self.to_align = to_align
        self.to_skip_bad = to_skip_bad
        self.imgs_shm = None
        self.y_boarders_shm = None
        self.x_boarders_shm = None
        self.raw_dataset, self.exposures, self.timestamps, self.img_shape, self.exclusion_boxes = \
            self.__load_raw_dataset(folder, non_linear, num_from_session)
        normalized_timestamps = [(item - min(self.timestamps)).total_seconds() for item in self.timestamps]
        new_timestamps = []
        first_ts = normalized_timestamps[0]
        for item in normalized_timestamps:
            if item - first_ts > 14 * 60 * 60:
                first_ts = item
            new_timestamps.append(item - first_ts)
        self.normalized_timestamps = np.array([item / max(new_timestamps) for item in new_timestamps])
        # self.normalized_timestamps = [(item - min(self.timestamps)).total_seconds() for item in self.timestamps]

        diff_timestamps = np.array(
            [(self.timestamps[i] - self.timestamps[i-1 if i-1 >= 0 else 0]
              ).total_seconds() for i in range(len(self.timestamps))])
        self.diff_timestamps = (diff_timestamps - np.min(diff_timestamps)
                                ) / (np.max(diff_timestamps) - np.min(diff_timestamps))
        # self.diff_timestamps = np.array(
        #     [(self.timestamps[i] - self.timestamps[i-1 if i-1 >= 0 else 0]
        #       ).total_seconds() for i in range(len(self.timestamps))])

    def __load_raw_dataset(self, folder, non_linear=False, num_from_session=0):
        raw_dataset, all_timestamps, all_exposures = self.crop_folder_on_the_fly(
            folder, non_linear=non_linear, num_from_session=num_from_session)
        img_shape = raw_dataset.shape[1:]
        exclusion_boxes = self.__load_exclusion_boxes(folder, img_shape)

        return raw_dataset, all_exposures, all_timestamps, img_shape, exclusion_boxes

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

    def crop_folder_on_the_fly(self, input_folder, non_linear, num_from_session=0):
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

        if num_from_session > 0:
            new_file_paths = []
            new_timestamps = []
            new_exposures = []
            add_number = 0
            for num in range(len(timestamps)):
                # new session if timestamp diss is more than 10 hours
                if timestamps[num] - timestamps[num-1 if num-1 >= 0 else 0] > datetime.timedelta(hours=10):
                    add_number = 0
                if add_number < num_from_session:
                    new_file_paths.append(file_paths[num])
                    new_timestamps.append(timestamps[num])
                    new_exposures.append(exposures[num])
                    add_number += 1
            file_paths = new_file_paths
            timestamps = new_timestamps
            exposures = new_exposures

        print("Loading and cropping images...")
        time.sleep(0.1)

        img, _, _ = load_func(file_paths[0], non_linear, True)
        mem_size = 1
        for n in (len(file_paths), *img.shape):
            mem_size = mem_size * n
        mem_size = mem_size * 4
        self.imgs_shm = SharedMemory(size=mem_size, create=True)
        images_shape = (len(file_paths), *img.shape[:2])
        imgs = np.ndarray(images_shape, dtype='float32', buffer=self.imgs_shm.buf)

        self.y_boarders_shm = SharedMemory(size=len(file_paths) * 2 * 2, create=True)
        y_boaredrs = np.ndarray((len(file_paths), 2), dtype='uint16', buffer=self.y_boarders_shm.buf)
        self.x_boarders_shm = SharedMemory(size=len(file_paths) * 2 * 2, create=True)
        x_boaredrs = np.ndarray((len(file_paths), 2), dtype='uint16', buffer=self.x_boarders_shm.buf)
        worker_num = min((os.cpu_count(), 4))
        processes = []
        progress_bar = tqdm.tqdm(total=len(file_paths))
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
                    progress_queue,
                    img,
                    self.to_align,
                    self.to_skip_bad
                ))
            )
        for proc in processes:
            proc.start()

        rejected_list = []
        for _ in range(len(file_paths)):
            res = progress_queue.get()
            num, rejected = res
            if rejected:
                rejected_list.append(num)

            progress_bar.update()

        for proc in processes:
            proc.join()
        progress_bar.close()

        if rejected_list:
            imgs = np.delete(imgs, rejected_list, axis=0)
            x_boaredrs = np.delete(x_boaredrs, rejected_list, axis=0)
            y_boaredrs = np.delete(y_boaredrs, rejected_list, axis=0)

        y_boarders = int(np.max(y_boaredrs[:, 0])), int(np.min(y_boaredrs[:, 1]))
        x_boarders = int(np.max(x_boaredrs[:, 0])), int(np.min(x_boaredrs[:, 1]))
        print(y_boarders, x_boarders)
        x_left, x_right = x_boarders
        y_top, y_bottom = y_boarders
        imgs = imgs[:, y_top: y_bottom, x_left: x_right]
        imgs = self.chop_imgs(imgs)
        print(imgs.shape)
        return imgs, timestamps, exposures

    def chop_imgs(self, imgs):
        shape = imgs[0].shape
        y_shape, x_shape = shape[:2]
        y_split_size = (y_shape - 2 * self.BOARDER_OFFSET) // self.Y_SPLITS
        x_split_size = (x_shape - 2 * self.BOARDER_OFFSET) // self.X_SPLITS
        imgs = imgs[
           :,
           :y_split_size * self.Y_SPLITS + 2 * self.BOARDER_OFFSET,
           :x_split_size * self.X_SPLITS + 2 * self.BOARDER_OFFSET,
        ]
        return imgs

    def gen_splits(self):
        shape = self.raw_dataset[0].shape
        y_shape, x_shape = shape[:2]
        y_split_size = (y_shape - 2 * self.BOARDER_OFFSET) // self.Y_SPLITS
        x_split_size = (x_shape - 2 * self.BOARDER_OFFSET) // self.X_SPLITS
        for y_num in range(self.Y_SPLITS):
            for x_num in range(self.X_SPLITS):
                y_offset = self.BOARDER_OFFSET + y_num * y_split_size
                x_offset = self.BOARDER_OFFSET + x_num * x_split_size
                not_aligned = self.raw_dataset[
                              :,
                              y_offset - self.BOARDER_OFFSET: y_offset + y_split_size + self.BOARDER_OFFSET,
                              x_offset - self.BOARDER_OFFSET: x_offset + x_split_size + self.BOARDER_OFFSET]
                if self.to_align:
                    aligned = np.ndarray((len(self.raw_dataset), y_split_size, x_split_size), dtype='float32')
                    aligned[0] = not_aligned[0, self.BOARDER_OFFSET: -self.BOARDER_OFFSET, self.BOARDER_OFFSET: -self.BOARDER_OFFSET]
                    bad = []
                    for num in range(1, len(not_aligned)):
                        try:
                            new_img, _ = aa.register(not_aligned[num], aligned[0], fill_value=0, min_area=9)
                            aligned[num] = new_img
                        except aa.MaxIterError:
                            if self.to_skip_bad:
                                bad.append(num)
                            else:
                                raise Exception(
                                    "Unable to make star alignment. Try to delete bad images or use --skip_bad key")
                    if bad:
                        aligned = np.delete(aligned, bad, axis=0)
                else:
                    aligned = not_aligned[:, self.BOARDER_OFFSET: -self.BOARDER_OFFSET, self.BOARDER_OFFSET: -self.BOARDER_OFFSET]

                yield aligned, y_offset, x_offset


class Dataset:
    ZERO_TOLERANCE = 100

    def __init__(self, source_data):

        if isinstance(source_data, SourceData):
            self.source_data = [source_data]
        else:
            self.source_data = source_data
        print("Raw image data loaded:")
        print(f"SHAPE: {[item.raw_dataset.shape for item in self.source_data]}")
        print(f"Used RAM: {sum([item.raw_dataset.itemsize * item.raw_dataset.size for item in self.source_data]) // (1024 * 1024)} Mb")

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
        dataset = self.source_data[dataset_idx].raw_dataset if dataset is None else dataset
        shrinked = np.copy(dataset[:, y:y+size, x:x+size])
        return shrinked

import astropy.io.fits
import cv2
import numpy as np


from xisf import XISF
from multiprocessing import Queue, cpu_count, Pool, Manager
from typing import Optional
from functools import partial

from backend.progress_bar import AbstractProgressBar
from logger.logger import get_logger
from backend.data_classes import SharedMemoryParams
from threading import Event
from backend.consuming_functions.measure_execution_time import measure_execution_time


PIXEL_TYPE = np.float32
logger = get_logger()


def debayer(img_data):
    res = np.array(cv2.cvtColor(img_data, cv2.COLOR_BayerBG2GRAY))
    res.reshape(img_data.shape[0], img_data.shape[1], 1)
    return res


def to_gray(img_data):
    return np.array(cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY))


def load_image_fits(filename: str, to_debayer: bool = False) -> np.ndarray:
    with astropy.io.fits.open(filename) as hdul:
        img_data = np.array(hdul[0].data)
        if len(img_data.shape) == 2:
            img_data = img_data.reshape(img_data.shape[0], img_data.shape[1], 1)
        if img_data.shape[0] in [1, 3]:
            img_data = np.swapaxes(img_data, 0, 2)
        if to_debayer and img_data.shape[2] == 1:
            img_data = np.array(debayer(img_data))
        img_data = img_data.reshape(img_data.shape[0], img_data.shape[1], 1)
        img_data = np.array(img_data)
        if img_data.shape[2] == 3:
            img_data = np.array(to_gray(img_data))
        # Normalize
        img_data = img_data.astype('float32')
        img_data /= 256 * 256 - 1
        img_data = img_data.astype(PIXEL_TYPE)
        img_data.shape = *img_data.shape[:2],
    return img_data


def load_image_xisf(filename: str, to_debayer: bool = False) -> np.ndarray:
    _ = to_debayer
    xisf = XISF(filename)
    img_data = xisf.read_image(0)
    img_data = np.array(img_data)
    if len(img_data.shape) == 2:
        img_data.shape = *img_data.shape, 1
    if img_data.shape[0] in [1, 3]:
        img_data = np.swapaxes(img_data, 0, 2)
    if img_data.shape[2] == 3:
        img_data = np.array(to_gray(img_data))
    img_data = img_data.astype(PIXEL_TYPE)
    if len(img_data.shape) == 2:
        img_data.shape = *img_data.shape[:2],
    return img_data


def load_image(file_path: str, to_debayer: bool = False) -> np.ndarray:
    if file_path.lower().endswith(".fits"):
        return load_image_fits(file_path, to_debayer)
    elif file_path.lower().endswith(".xisf"):
        return load_image_xisf(file_path, to_debayer)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def load_worker(indexes: list[int], file_list: list[str], shm_params: SharedMemoryParams, progress_queue: Queue,
                to_debayer: bool = False, stop_queue: Optional[Queue] = None) -> None:
    try:
        imgs = np.memmap(shm_params.shm_name, dtype=PIXEL_TYPE, mode='r+', shape=shm_params.shm_shape)
        for img_idx in indexes:
            if stop_queue and not stop_queue.empty():
                break
            img_data = load_image(file_list[img_idx], to_debayer)
            imgs[img_idx] = img_data
            progress_queue.put(img_idx)
            imgs.flush()
    except:
        import traceback
        traceback.print_exc()


@measure_execution_time
def load_images(file_list: list[str], shm_params: SharedMemoryParams, to_debayer: bool = False,
                progress_bar: Optional[AbstractProgressBar] = None, stop_event: Optional[Event] = None) -> None:
    available_cpus = cpu_count()
    used_cpus = min(available_cpus, len(file_list))
    with (Pool(processes=used_cpus) as pool):
        m = Manager()
        progress_queue = m.Queue()
        stop_queue = m.Queue(maxsize=1)
        results = pool.map_async(
            partial(
                load_worker,
                file_list=file_list,
                shm_params=shm_params,
                to_debayer=to_debayer,
                progress_queue=progress_queue,
                stop_queue=stop_queue
            ),
            np.array_split(np.arange(len(file_list)), used_cpus))
        if progress_bar is not None:
            progress_bar.set_total(len(file_list))
            for _ in range(len(file_list)):
                if stop_event is not None and stop_event.is_set():
                    stop_queue.put(True)
                    break
                img_idx = progress_queue.get()
                logger.log.debug(f"Loaded image '{file_list[img_idx]}' at index '{img_idx}'")
                progress_bar.update()
            progress_bar.complete()
        results.get()
        pool.close()
        pool.join()

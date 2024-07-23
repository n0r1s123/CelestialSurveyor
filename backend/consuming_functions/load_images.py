import astropy.io.fits
import cv2
import numpy as np
import traceback

from xisf import XISF
from multiprocessing import Queue, cpu_count, Pool, Manager
from typing import Optional
from functools import partial

from backend.progress_bar import AbstractProgressBar
from logger.logger import get_logger
from backend.data_classes import SharedMemoryParams
from threading import Event
from backend.consuming_functions.measure_execution_time import measure_execution_time
from logging.handlers import QueueHandler


PIXEL_TYPE = np.float32
logger = get_logger()


def debayer(img_data: np.ndarray) -> np.ndarray:
    """
    Perform debayering on the input image data.

    Args:
        img_data: Image data to be debayered.

    Returns:
        np.ndarray: Debayered image data.
    """
    res = np.array(cv2.cvtColor(img_data, cv2.COLOR_BayerBG2GRAY))
    res.reshape(img_data.shape[0], img_data.shape[1], 1)
    return res


def to_gray(img_data: np.ndarray) -> np.ndarray:
    """
    Convert the input image data to grayscale.

    Args:
        img_data (np.ndarray): The input image data to be converted to grayscale.

    Returns:
        np.ndarray: The grayscale image data.
    """
    return np.array(cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY))


def load_image_fits(filename: str, to_debayer: bool = False) -> np.ndarray:
    """
    Load image data from a FIT(S) file and optionally debayer it.

    Args:
        filename (str): The path to the FIT(S) file.
        to_debayer (bool): Whether to perform debayering.

    Returns:
        np.ndarray: numpy array containing the image data converted to grayscale.
    """
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
    """
    Load image data from a XISF file and optionally debayer it.

    Args:
        filename (str): The path to the FITS file.
        to_debayer (bool): Whether to perform debayering.

    Returns:
        np.ndarray: numpy array containing the image data converted to grayscale.
    """
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
    """
    Load an image file and optionally perform debayering.

    Args:
        file_path (str): The path to the image file.
        to_debayer (bool): Whether to perform debayering.

    Returns:
        np.ndarray: numpy array containing the image data converted to grayscale.
    """
    if file_path.lower().endswith(".fits") or file_path.lower().endswith(".fit"):
        return load_image_fits(file_path, to_debayer)
    elif file_path.lower().endswith(".xisf"):
        return load_image_xisf(file_path, to_debayer)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def load_worker(indexes: list[int], file_list: list[str], shm_params: SharedMemoryParams, progress_queue: Queue,
                to_debayer: bool = False, stop_queue: Optional[Queue] = None, log_queue: Optional[Queue] = None
                ) -> None:
    """
    Load images into shared memory based on the provided indexes and file list.

    Args:
        indexes (list[int]): List of indexes specifying which images to load.
        file_list (list[str]): List of file paths for the images.
        shm_params (SharedMemoryParams): Shared memory parameters for loading images.
        progress_queue (Queue): Queue for reporting progress.
        to_debayer (bool, optional): Whether to perform debayering. Defaults to False.
        stop_queue (Queue, optional): Queue for stopping the loading process. Defaults to None.
        log_queue (Queue, optional): Queue for logging messages. Defaults to None.

    Returns:
        None
    """
    handler = QueueHandler(log_queue)
    logger.log.addHandler(handler)
    logger.log.debug(f"Load worker started with {len(indexes)} images")
    logger.log.debug(f"Shared memory parameters: {shm_params}")
    try:
        imgs = np.memmap(shm_params.shm_name, dtype=PIXEL_TYPE, mode='r+', shape=shm_params.shm_shape)
        for img_idx in indexes:
            if stop_queue and not stop_queue.empty():
                logger.log.debug("Load images worker detected stop event. Stopping.")
                break
            img_data = load_image(file_list[img_idx], to_debayer)
            imgs[img_idx] = img_data
            progress_queue.put(img_idx)

    except Exception:
        logger.log.error(f"Load worker failed due to the following error:\n{traceback.format_exc()}")
        stop_queue.put("ERROR")
        raise


@measure_execution_time
def load_images(file_list: list[str], shm_params: SharedMemoryParams, to_debayer: bool = False,
                progress_bar: Optional[AbstractProgressBar] = None, stop_event: Optional[Event] = None):
    """
    Load images from the provided file list into shared memory using multiple workers.

    Args:
        file_list (list[str]): List of file paths for the images to load.
        shm_params (SharedMemoryParams): Shared memory parameters for loading images.
        to_debayer (bool, optional): Whether to perform debayering. Defaults to False.
        progress_bar (Optional[AbstractProgressBar], optional): Progress bar for tracking the loading progress.
            Defaults to None.
        stop_event (Optional[Event], optional): Event to signal stopping the loading process. Defaults to None.

    Returns:
        None
    """
    available_cpus = cpu_count() - 1
    used_cpus = min(available_cpus, len(file_list))
    logger.log.debug(f"Number of CPUs to be used for loading images: {used_cpus}")
    with (Pool(processes=used_cpus) as pool):
        m = Manager()
        progress_queue = m.Queue()
        log_queue = m.Queue()
        logger.start_process_listener(log_queue)
        stop_queue = m.Queue()
        logger.log.debug(f"Starting loading images with {used_cpus} workers")
        results = pool.map_async(
            partial(
                load_worker,
                file_list=file_list,
                shm_params=shm_params,
                to_debayer=to_debayer,
                progress_queue=progress_queue,
                stop_queue=stop_queue,
                log_queue=log_queue
            ),
            np.array_split(np.arange(len(file_list)), used_cpus))
        if progress_bar is not None:
            progress_bar.set_total(len(file_list))
            for _ in range(len(file_list)):
                if stop_event is not None and stop_event.is_set():
                    stop_queue.put(True)
                    logger.log.debug("Stop event triggered")
                    break

                got_result = False
                while not got_result:
                    if not progress_queue.empty():
                        progress_queue.get()
                        logger.log.debug("Got a result from the progress queue")
                        got_result = True
                    if not stop_queue.empty():
                        logger.log.debug("Detected error from workers. Stopping.")
                        break
                if not stop_queue.empty():
                    break
                progress_bar.update()
            progress_bar.complete()
        results.get()
        pool.close()
        pool.join()
        logger.log.debug(f"Load images pool stopped.")
        logger.stop_process_listener()

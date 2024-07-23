import numpy as np
import traceback

from auto_stretch.stretch import Stretch
from functools import partial
from logging.handlers import QueueHandler
from multiprocessing import Queue, cpu_count, Pool, Manager
from threading import Event
from typing import Optional

from backend.progress_bar import AbstractProgressBar
from logger.logger import get_logger
from backend.data_classes import SharedMemoryParams
from backend.consuming_functions.measure_execution_time import measure_execution_time


logger = get_logger()


@measure_execution_time
def stretch_images(shm_params: SharedMemoryParams, progress_bar: Optional[AbstractProgressBar] = None,
                   stop_event: Optional[Event] = None) -> None:
    """
    Stretch images stored in shared memory in parallel using multiprocessing.

    Args:
        shm_params (SharedMemoryParams): Shared memory parameters for the images.
        progress_bar (Optional[AbstractProgressBar]): Progress bar to track the stretching progress.
        stop_event (Optional[Event]): Event to stop the stretching process.

    Returns:
        None
    """
    available_cpus = cpu_count() - 1
    frames_num = shm_params.shm_shape[0]
    used_cpus = min(available_cpus, frames_num)
    logger.log.debug(f"Number of CPUs to be used for loading images: {used_cpus}")
    with Pool(processes=used_cpus) as pool:
        m = Manager()
        progress_queue = m.Queue()
        stop_queue = m.Queue(maxsize=1)
        log_queue = m.Queue()
        logger.start_process_listener(log_queue)
        logger.log.debug(f"Starting stretching images with {used_cpus} workers")
        results = pool.map_async(
            partial(stretch_worker, shm_params=shm_params, progress_queue=progress_queue, stop_queue=stop_queue,
                    log_queue=log_queue),
            np.array_split(np.arange(frames_num), used_cpus))
        if progress_bar is not None:
            progress_bar.set_total(frames_num)
            for _ in range(frames_num):
                if stop_event is not None and stop_event.is_set():
                    stop_queue.put(True)
                    logger.log.debug("Stop event triggered")
                    break
                got_result = False
                while not got_result:
                    if not progress_queue.empty():
                        progress_queue.get()
                        got_result = True
                        logger.log.debug("Got a result from the progress queue")
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
        logger.log.debug(f"Plate solve pool stopped.")
        logger.stop_process_listener()


def stretch_worker(img_indexes: list[int], shm_params: SharedMemoryParams, progress_queue: Queue,
                   stop_queue: Optional[Queue] = None, log_queue: Optional[Queue] = None) -> None:
    """
    Worker function to stretch images with the provided indexes in shared memory.

    Args:
        img_indexes (list[int]): List of image indexes to stretch.
        shm_params (SharedMemoryParams): Shared memory parameters for images.
        progress_queue (Queue): Queue for reporting progress.
        stop_queue (Optional[Queue], optional): Queue for stopping the worker process. Defaults to None.
        log_queue (Optional[Queue], optional): Queue for logging messages. Defaults to None.

    Returns:
        None
    """
    handler = QueueHandler(log_queue)
    logger.log.addHandler(handler)
    logger.log.debug(f"Load worker started with {len(img_indexes)} images")
    logger.log.debug(f"Shared memory parameters: {shm_params}")
    try:
        imgs = np.memmap(shm_params.shm_name, dtype=shm_params.shm_dtype, mode='r+', shape=shm_params.shm_shape)
        for img_idx in img_indexes:
            if stop_queue is not None and not stop_queue.empty():
                logger.log.debug("Plate solve worker detected stop event. Stopping.")
                break
            img = imgs[img_idx, shm_params.y_slice, shm_params.x_slice]
            imgs[img_idx, shm_params.y_slice, shm_params.x_slice] = Stretch().stretch(img)
            progress_queue.put(img_idx)
            imgs.flush()
    except Exception:
        logger.log.error(f"Stretch worker failed due to the following error:\n{traceback.format_exc()}")
        stop_queue.put("ERROR")
        raise

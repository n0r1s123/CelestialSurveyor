import numpy as np
from auto_stretch.stretch import Stretch
from multiprocessing import Queue, cpu_count, Pool, Manager
from typing import Optional
from functools import partial
import traceback
from logging.handlers import QueueHandler

from backend.progress_bar import AbstractProgressBar
from logger.logger import get_logger
from backend.data_classes import SharedMemoryParams
from threading import Event
from backend.consuming_functions.measure_execution_time import measure_execution_time

logger = get_logger()


@measure_execution_time
def stretch_images(shm_params: SharedMemoryParams,
                   progress_bar: Optional[AbstractProgressBar] = None, stop_event: Optional[Event] = None) -> None:

    available_cpus = cpu_count()
    frames_num = shm_params.shm_shape[0] - 1
    used_cpus = min(available_cpus, frames_num)
    with Pool(processes=used_cpus) as pool:
        m = Manager()
        progress_queue = m.Queue()
        stop_queue = m.Queue(maxsize=1)
        log_queue = m.Queue()
        logger.start_process_listener(log_queue)
        results = pool.map_async(
            partial(stretch_worker, shm_params=shm_params, progress_queue=progress_queue, stop_queue=stop_queue,
                    log_queue=log_queue),
            np.array_split(np.arange(frames_num + 1), used_cpus))

        if progress_bar is not None:
            progress_bar.set_total(frames_num)
            for _ in range(frames_num):
                if stop_event is not None and stop_event.is_set():
                    stop_queue.put(True)
                    break
                got_result = False
                while not got_result:
                    if not progress_queue.empty():
                        progress_queue.get()
                        got_result = True
                    if not stop_queue.empty():
                        break
                if not stop_queue.empty():
                    break
                progress_bar.update()
            progress_bar.complete()
        results.get()
        pool.close()
        pool.join()
        logger.stop_process_listener()


def stretch_worker(img_indexes: list[int], shm_params: SharedMemoryParams, progress_queue: Queue,
                   stop_queue: Optional[Queue] = None, log_queue: Optional[Queue] = None) -> None:
    handler = QueueHandler(log_queue)
    logger.log.addHandler(handler)
    logger.log.debug(f"Load worker started with {len(img_indexes)} images")
    logger.log.debug(f"Shared memory parameters: {shm_params}")
    try:
        imgs = np.memmap(shm_params.shm_name, dtype=shm_params.shm_dtype, mode='r+', shape=shm_params.shm_shape)
        for img_idx in img_indexes:
            if stop_queue is not None and not stop_queue.empty():
                break
            img = imgs[img_idx, shm_params.y_slice, shm_params.x_slice]
            imgs[img_idx, shm_params.y_slice, shm_params.x_slice] = Stretch().stretch(img)
            progress_queue.put(img_idx)
            imgs.flush()
    except Exception:
        logger.log.error(f"Stretch worker failed due to the following error:\n{traceback.format_exc()}")
        stop_queue.put("ERROR")
        raise

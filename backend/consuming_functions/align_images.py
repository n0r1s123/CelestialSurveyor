import numpy as np
import traceback

from logging.handlers import QueueHandler
from multiprocessing import Queue, cpu_count, Pool, Manager
from typing import Optional
from functools import partial

from backend.progress_bar import AbstractProgressBar
from logger.logger import get_logger
from backend.data_classes import SharedMemoryParams
from astropy.wcs import WCS
from reproject import reproject_interp
from threading import Event
from backend.consuming_functions.measure_execution_time import measure_execution_time


logger = get_logger()


@measure_execution_time
def align_images_wcs(shm_params: SharedMemoryParams, all_wcses: list[WCS],
                     progress_bar: Optional[AbstractProgressBar] = None, stop_event: Optional[Event] = None
                     ) -> tuple[list[bool], np.ndarray]:

    available_cpus = min(cpu_count(), 8)
    frames_num = shm_params.shm_shape[0]
    used_cpus = min(available_cpus, frames_num)
    with Pool(processes=used_cpus) as pool:
        m = Manager()
        progress_queue = m.Queue()
        stop_queue = m.Queue(maxsize=1)
        log_queue = m.Queue()
        logger.start_process_listener(log_queue)
        results = pool.map_async(
            partial(align_wcs_worker, shm_params=shm_params, progress_queue=progress_queue, ref_wcs=all_wcses[0],
                    all_wcses=all_wcses, stop_queue=stop_queue, log_queue=log_queue),
            np.array_split(np.arange(frames_num), used_cpus))

        if progress_bar is not None:
            progress_bar.set_total(frames_num)
            for _ in range(frames_num):
                if stop_event is not None and stop_event.is_set():
                    stop_queue.put(True)
                    break
                img_idx = progress_queue.get()
                logger.log.debug(f"Aligned image at index '{img_idx}'")
                progress_bar.update()
            progress_bar.complete()

        res = results.get()
        idxs = []
        successes = []
        footprints = []
        for idx, success, footprint in res:
            idxs.extend(idx)
            successes.extend(success)
            footprints.extend(footprint)
        res = list(zip(idxs, successes, footprints))
        res.sort(key=lambda item: item[0])
        success_map = [item[1] for item in res]
        footprint_map = np.array([item[2] for item in res])
        pool.close()
        pool.join()
        logger.stop_process_listener()
    return success_map, footprint_map


def align_wcs_worker(img_indexes: list[int], shm_params: SharedMemoryParams, progress_queue: Queue, ref_wcs: WCS,
                     all_wcses: list[WCS], stop_queue: Optional[Queue] = None, log_queue: Optional[Queue] = None
                     ) -> tuple[list[int], list[bool], list[np.ndarray]]:
    imgs = np.memmap(shm_params.shm_name, dtype=shm_params.shm_dtype, mode='r+', shape=shm_params.shm_shape)
    footprints = []
    successes = []
    handler = QueueHandler(log_queue)
    logger.log.addHandler(handler)
    logger.log.debug(f"Align worker started with {len(img_indexes)} images")
    logger.log.debug(f"Shared memory parameters: {shm_params}")
    for img_idx in img_indexes:
        if stop_queue is not None and not stop_queue.empty():
            break
        try:
            # TODO: Think how to save memory
            _, footprint = reproject_interp(
                (np.reshape(np.copy(imgs[img_idx]), shm_params.shm_shape[1:3]),
                 all_wcses[img_idx]),
                ref_wcs,
                shape_out=shm_params.shm_shape[1:3],
                output_array=imgs[img_idx],
            )
            imgs.flush()
        except Exception:
            footprint = np.ones(shm_params.shm_shape[1:], dtype=bool)
            success = False
            logger.log.error(f"Align worker failed to process image at index "
                             f"'{img_idx}' due to the following error:\n{traceback.format_exc()}")
        else:
            success = True
            footprint = 1 - footprint
            footprint = np.array(footprint, dtype=bool)
        footprints.append(footprint)
        successes.append(success)

        progress_queue.put(img_idx)
    logger.log.removeHandler(handler)
    return img_indexes, successes, footprints

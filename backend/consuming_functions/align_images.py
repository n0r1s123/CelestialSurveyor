import numpy as np
import traceback

from functools import partial
from logging.handlers import QueueHandler
from multiprocessing import Manager, Pool, Queue, cpu_count
from typing import Optional
from threading import Event

from astropy.wcs import WCS
from backend.consuming_functions.measure_execution_time import measure_execution_time
from backend.data_classes import SharedMemoryParams
from backend.progress_bar import AbstractProgressBar
from logger.logger import get_logger
from reproject import reproject_interp


logger = get_logger()


@measure_execution_time
def align_images_wcs(shm_params: SharedMemoryParams, all_wcs: list[WCS],
                     progress_bar: Optional[AbstractProgressBar] = None, stop_event: Optional[Event] = None
                     ) -> tuple[list[bool], np.ndarray]:
    """
    Align images with World Coordinate System (WCS).

    Args:
        shm_params (SharedMemoryParams): Shared memory parameters where the images are stored.
        all_wcs (list[WCS]): List of WCS objects to be used for alignment.
        progress_bar (Optional[AbstractProgressBar]): Progress bar object.
        stop_event (Optional[Event]): Stop event object used to stop the child processes.

    Returns:
        tuple[list[bool], np.ndarray]: Tuple containing a list of success flags and a numpy array of footprints.
    """
    available_cpus = min(cpu_count(), 4)
    frames_num = shm_params.shm_shape[0]
    used_cpus = min(available_cpus, frames_num)
    logger.log.debug(f"Number of CPUs to be used for alignment: {used_cpus}")
    with Pool(processes=used_cpus) as pool:
        m = Manager()
        progress_queue = m.Queue()
        stop_queue = m.Queue(maxsize=1)
        log_queue = m.Queue()
        logger.start_process_listener(log_queue)
        logger.log.debug(f"Starting alignment with {used_cpus} workers")
        results = pool.map_async(
            partial(align_wcs_worker, shm_params=shm_params, progress_queue=progress_queue, ref_wcs=all_wcs[0],
                    all_wcses=all_wcs, stop_queue=stop_queue, log_queue=log_queue),
            np.array_split(np.arange(frames_num), used_cpus))
        if progress_bar is not None:
            progress_bar.set_total(frames_num)
            for _ in range(frames_num):
                if stop_event is not None and stop_event.is_set():
                    logger.log.debug("Stop event triggered")
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
        logger.log.debug(f"Alignment finished. Success map: {successes}")
        res = list(zip(idxs, successes, footprints))
        res.sort(key=lambda item: item[0])
        success_map = [item[1] for item in res]
        footprint_map = np.array([item[2] for item in res])
        pool.close()
        pool.join()
        logger.log.debug(f"Alignment pool stopped.")
        logger.stop_process_listener()
    return success_map, footprint_map


def align_wcs_worker(img_indexes: list[int], shm_params: SharedMemoryParams, progress_queue: Queue, ref_wcs: WCS,
                     all_wcses: list[WCS], stop_queue: Optional[Queue] = None, log_queue: Optional[Queue] = None
                     ) -> tuple[list[int], list[bool], list[np.ndarray]]:
    """
    Worker function for aligning images basing on the WCS information.

    Args:
        img_indexes (list[int]): List of image indexes to align within this worker.
        shm_params (SharedMemoryParams): Shared memory parameters where the images are stored.
        progress_queue (Queue): Queue to report progress.
        ref_wcs (WCS): Reference WCS information to be used for alignment.
        all_wcses (list[WCS]): List of all WCS information.
        stop_queue (Optional[Queue], optional): Queue to stop the process. Defaults to None.
        log_queue (Optional[Queue], optional): Queue for logging. Defaults to None.

    Returns:
        tuple[list[int], list[bool], list[np.ndarray]]: A tuple containing aligned image indexes, success status list,
        and footprints.
    """
    imgs = np.memmap(shm_params.shm_name, dtype=shm_params.shm_dtype, mode='r+', shape=shm_params.shm_shape)
    footprints = []
    successes = []
    handler = QueueHandler(log_queue)
    logger.log.addHandler(handler)
    logger.log.debug(f"Align worker started with {len(img_indexes)} images")
    logger.log.debug(f"Shared memory parameters: {shm_params}")
    for img_idx in img_indexes:
        too_far = False
        if stop_queue is not None and not stop_queue.empty():
            logger.log.debug("Align worker detected stop event. Stopping.")
            break

        try:
            distance = get_centre_distance(imgs[0].shape, ref_wcs, all_wcses[img_idx])
            if distance > 0.1 * min(imgs[0].shape[:2]):
                too_far = True
                logger.log.warning(f"Align worker detected that image at index '{img_idx}' is too far "
                                   f"from the reference solution.  Pixel distance: {distance}. Excluding this image")
            else:
                _, footprint = reproject_interp(
                    (np.reshape(np.copy(imgs[img_idx]), shm_params.shm_shape[1:3]),
                     all_wcses[img_idx]),
                    ref_wcs,
                    shape_out=shm_params.shm_shape[1:3],
                    output_array=imgs[img_idx],
                )
                imgs.flush()
        except Exception:
            # if an error occurs, assume that the image is not aligned, and mark it as such. In this case alignment
            # process is not considered to be failed, failed image will not be used in the next steps
            footprint = np.ones(shm_params.shm_shape[1:], dtype=bool)
            success = False
            logger.log.error(f"Align worker failed to process image at index "
                             f"'{img_idx}' due to the following error:\n{traceback.format_exc()}")
        else:
            if too_far:
                footprint = np.ones(shm_params.shm_shape[1:], dtype=bool)
                success = False
            else:
                success = True
                # this line is needed to be consistent with the legacy code which does image cropping
                footprint = 1 - footprint
                footprint = np.array(footprint, dtype=bool)
        footprints.append(footprint)
        successes.append(success)

        progress_queue.put(img_idx)
    logger.log.removeHandler(handler)
    return img_indexes, successes, footprints


def get_centre_distance(img_shape: tuple, wcs1: WCS, wcs2: WCS):
    """
    Calculate the distance between the centers of two images.

    Args:
        img_shape (tuple[int, int]): Shape of the image.
        wcs1 (WCS): WCS information of the first image.
        wcs2 (WCS): WCS information of the second image.

    Returns:
        float: The distance between the centers of the two images.
    """
    ref_center_x, ref_center_y = img_shape[1] / 2, img_shape[0] / 2
    center_coordinates = wcs2.pixel_to_world(ref_center_x, ref_center_y)
    second_centre_on_ref_image = wcs1.world_to_pixel(center_coordinates)
    logger.log.debug(f"Second centre: {second_centre_on_ref_image}")
    second_center_x, second_center_y = second_centre_on_ref_image
    second_center_x = int(second_center_x)
    second_center_y = int(second_center_y)
    distance = np.sqrt((ref_center_x - second_center_x) ** 2 + (ref_center_y - second_center_y) ** 2)
    logger.log.debug(f"Distance: {distance}")
    return distance

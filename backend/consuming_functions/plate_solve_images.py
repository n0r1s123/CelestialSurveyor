import numpy as np
import traceback
import twirl

from astropy import units as u
from astropy.wcs import WCS
from functools import partial
from logging.handlers import QueueHandler
from multiprocessing import Queue, cpu_count, Pool, Manager
from threading import Event
from typing import Optional

from backend.progress_bar import AbstractProgressBar
from backend.data_classes import Header
from logger.logger import get_logger
from backend.data_classes import SharedMemoryParams
from backend.consuming_functions.measure_execution_time import measure_execution_time


logger = get_logger()


def plate_solve_image(image: np.ndarray, header: Header,
                      sky_coord: Optional[np.ndarray] = None) -> tuple[WCS, np.ndarray]:
    """
    Plate solves the image basing on the provided GAIA data. If this data is not available - it will be requested.

    Args:
        image (np.ndarray): The image to be plate solved.
        header (Header): The header information related to the image.
        sky_coord (Optional[np.ndarray]): Source information for the current FOV got from GAIA. Defaults to None.

    Returns:
        tuple[WCS, np.ndarray]: The plate solved WCS and sky coordinates.
    """
    header_data = header.solve_data
    pixel = header_data.pixel_scale * u.arcsec  # known pixel scale
    img = np.copy(image)
    img = np.reshape(img, (img.shape[0], img.shape[1]))
    shape = img.shape
    fov = np.min(shape[:2]) * pixel.to(u.deg)
    if sky_coord is None:
        sky_coord = twirl.gaia_radecs(header_data.sky_coord, fov)[0:200]
        sky_coord = twirl.geometry.sparsify(sky_coord, 0.1)
        sky_coord = sky_coord[:25]
    top_left_corner = (slice(None, img.shape[0] // 2), slice(None, img.shape[1] // 2), (0, 0))
    bottom_left_corner = (slice(img.shape[0] // 2, None), slice(None, img.shape[1] // 2), (img.shape[0]//2, 0))
    top_right_corner = (slice(None, img.shape[0] // 2), slice(img.shape[1] // 2, None), (0, img.shape[1]//2))
    bottom_right_corner = (slice(img.shape[0] // 2, None), slice(img.shape[1] // 2, None),
                           (img.shape[0]//2, img.shape[1]//2))
    corners = [top_left_corner, bottom_left_corner, top_right_corner, bottom_right_corner]
    all_corner_peaks = []
    for y_slice, x_slice, (y_offset, x_offset) in corners:
        peak_cos = twirl.find_peaks(img[y_slice, x_slice], threshold=1)[0:200]
        corner_peaks = []
        for x, y in peak_cos:
            y += y_offset
            x += x_offset
            dist_from_center_x = x - shape[1] // 2
            dist_from_center_y = y - shape[0] // 2
            if np.sqrt(dist_from_center_x**2 + dist_from_center_y**2) < min(shape) // 2:
                corner_peaks.append([x, y])
        all_corner_peaks.extend(corner_peaks[:8])
    all_corner_peaks = np.array(all_corner_peaks)
    wcs = twirl.compute_wcs(all_corner_peaks, sky_coord, asterism=4)
    return wcs, sky_coord


@measure_execution_time
def plate_solve(shm_params: SharedMemoryParams, headers: list[Header],
                progress_bar: Optional[AbstractProgressBar] = None,
                stop_event: Optional[Event] = None) -> list[WCS]:
    """
    Plate solving images stored in shared memory.

    Args:
        shm_params (SharedMemoryParams): Shared memory parameters for accessing image data.
        headers (list[Header]): List of image headers.
        progress_bar (Optional[AbstractProgressBar]): Progress bar for tracking plate solving progress.
        stop_event (Optional[Event]): Event for stopping plate solving process.

    Returns:
        list[WCS]: List of plate solved WCS coordinates.
    """
    logger.log.info("Plate solving...")
    imgs = np.memmap(shm_params.shm_name, dtype=shm_params.shm_dtype, mode='r+', shape=shm_params.shm_shape)
    # get reference stars from GAIA for the first image's FOV
    _, reference_stars = plate_solve_image(imgs[0], headers[0])
    available_cpus = cpu_count()
    frames_num = shm_params.shm_shape[0]
    used_cpus = min(available_cpus, frames_num)
    logger.log.debug(f"Number of CPUs to be used for loading images: {used_cpus}")
    m = Manager()
    progress_queue = m.Queue()
    log_queue = m.Queue()
    logger.start_process_listener(log_queue)
    stop_queue = m.Queue(maxsize=1)
    with Pool(processes=used_cpus) as pool:
        logger.log.debug(f"Starting loading images with {used_cpus} workers")
        results = pool.map_async(
            partial(plate_solve_worker, shm_params=shm_params, progress_queue=progress_queue,
                    reference_stars=reference_stars, header=headers[0], stop_queue=stop_queue, log_queue=log_queue),
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
                        logger.log.debug("Got a result from the progress queue")
                        got_result = True
                    if not stop_queue.empty():
                        logger.log.debug("Detected error from workers. Stopping.")
                        break
                if not stop_queue.empty():
                    break
                progress_bar.update()
            progress_bar.complete()
        res = results.get()
    pool.close()
    pool.join()
    logger.log.debug(f"Plate solve pool stopped.")
    logger.stop_process_listener()
    new_res = []
    for item in res:
        new_res.extend(item)
    new_res.sort(key=lambda x: x[0])
    return [item[1] for item in new_res]


def plate_solve_worker(img_indexes: list[int], header: Header, shm_params: SharedMemoryParams,
                       reference_stars: np.ndarray, progress_queue: Queue, stop_queue: Optional[Queue] = None,
                       log_queue: Optional[Queue] = None) -> list[tuple[int, WCS]]:
    handler = QueueHandler(log_queue)
    logger.log.addHandler(handler)
    logger.log.debug(f"Load worker started with {len(img_indexes)} images")
    logger.log.debug(f"Shared memory parameters: {shm_params}")
    try:
        imgs = np.memmap(shm_params.shm_name, dtype=shm_params.shm_dtype, mode='r', shape=shm_params.shm_shape)
        res = []
        for img_idx in img_indexes:
            if stop_queue is not None and not stop_queue.empty():
                logger.log.debug("Plate solve worker detected stop event. Stopping.")
                break
            img = imgs[img_idx]
            wcs, _ = plate_solve_image(img, header, reference_stars)
            progress_queue.put(img_idx)
            res.append((img_idx, wcs))
    except Exception:
        logger.log.error(f"Plate solve worker failed due to the following error:\n{traceback.format_exc()}")
        stop_queue.put("ERROR")
        raise
    return res

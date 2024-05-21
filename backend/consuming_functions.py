import time
import astroalign as aa
import astropy.io.fits
import cv2
import multiprocessing
import numpy as np


from auto_stretch.stretch import Stretch
from datetime import datetime
from decimal import Decimal
from xisf import XISF
from multiprocessing import Queue, cpu_count, Pool
from typing import Optional
from multiprocessing.shared_memory import SharedMemory
from functools import partial
from astropy import units as u
from astropy.coordinates import SkyCoord

from backend.progress_bar import AbstractProgressBar
from backend.data_classes import SolveData, SiteLocation, Header
from logger.logger import get_logger
from backend.data_classes import SharedMemoryParams
import twirl
from astropy.wcs import WCS
from reproject import reproject_interp
from photutils.aperture import CircularAperture


logger = get_logger()

PIXEL_TYPE = "float32"
SECONDARY_ALIGNMENT_OFFSET = 20

ctx = multiprocessing.get_context()
# ctx.set_start_method("spawn", force=True)


def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.log.info(f"Execution time of {func.__name__}: {execution_time} seconds")
        return result
    return wrapper


def __get_datetime_from_str(date_str: str) -> datetime:
    """
    Convert a string date representation to a datetime object.

    Args:
        date_str (str): The string containing the date in the format "%Y-%m-%dT%H:%M:%S.%f".

    Returns:
        datetime: A datetime object representing the parsed date from the input string.
    """
    try:
        res = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        res = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
    return res


def load_header(filename: str, progress_queue: Optional[Queue] = None) -> Header:
    if filename.lower().endswith(".xisf"):
        header = load_header_xisf(filename)
    elif filename.lower().endswith(".fit") or filename.lower().endswith(".fits"):
        header = load_header_fits(filename)
    else:
        raise ValueError("File type not supported. Supported file types: .xisf, .fit, .fits")

    if progress_queue is not None:
        progress_queue.put(True)

    return header


def load_header_fits(filename: str) -> Header:
    # Open the FITS file
    with astropy.io.fits.open(filename) as hdul:
        # Get the header of the first HDU
        header = hdul[0].header

        # Extract the exposure time from the header
        exposure = Decimal(header['EXPTIME'])

        # Extract the timestamp from the header and convert it to datetime
        timestamp = __get_datetime_from_str(header['DATE-OBS'])

        # Extract the right ascension (RA) from the header
        ra = Decimal(header['RA'])

        # Extract the declination (DEC) from the header
        dec = Decimal(header['DEC'])

        # Extract the pixel scale from the header. If not present, calculate it from focal length and pixel size.
        pixel_scale = header.get('SCALE')
        if pixel_scale is not None:
            pixel_scale = Decimal(pixel_scale)
        else:
            focal_len = header.get('FOCALLEN')
            pixel_size = header.get('XPIXSZ')
            if focal_len is not None and pixel_size is not None:
                focal_len = Decimal(focal_len)
                pixel_size = Decimal(pixel_size)
                pixel_scale = (pixel_size / focal_len) * Decimal(206.265)
            else:
                raise ValueError("Pixel scale information is not present in FITS header")
        # Create a SolveData object with the extracted RA, DEC and pixel scale
        plate_solve_data = SolveData(SkyCoord(ra, dec, unit=["deg", "deg"]), pixel_scale)

        # Extract the latitude and longitude from the header
        lat = Decimal(header.get("SITELAT", 0))
        long = Decimal(header.get("SITELONG", 0))

        # Create a SiteLocation object with the extracted latitude and longitude
        site_location = SiteLocation(lat=lat, long=long)

        # Create a Header object with the extracted information
        header = Header(filename, exposure, timestamp,  site_location, plate_solve_data)

        # Return the Header object
        return header

@measure_execution_time
def load_headers(filenames: list[str], progress_bar: Optional[AbstractProgressBar] = None) -> list[Header]:
    # Load headers in parallel with multiprocessing.Pool
    available_cpu = min(cpu_count(), len(filenames))
    with Pool(available_cpu) as pool:
        m = ctx.Manager()
        progress_queue = m.Queue()
        results = pool.map_async(partial(load_header, progress_queue=progress_queue), filenames)

        if progress_bar is not None:
            progress_bar.set_total(len(filenames))
            for _ in range(len(filenames)):
                img_idx = progress_queue.get()
                logger.log.debug(f"Loaded image '{filenames[img_idx]}' at index '{img_idx}'")
                progress_bar.update()
            progress_bar.complete()
        headers = results.get()
        pool.close()
        pool.join()
        return headers


def load_header_xisf(filename: str) -> Header:
    # Initialize XISF object
    xisf = XISF(filename)
    # Get the metadata of the first image in the XISF file
    img_meta = xisf.get_images_metadata()[0]

    header = img_meta["FITSKeywords"]

    # Extract the timestamp from the FITS header
    timestamp = __get_datetime_from_str(header["DATE-OBS"][0]['value'])

    # Extract the exposure time from the FITS header
    exposure = Decimal(header["EXPTIME"][0]['value'])
    # Extract the right ascension (RA) from the FITS header
    ra = header.get("RA")
    if ra is not None:
        ra = Decimal(ra[0]['value'])
    dec = header.get("DEC")
    if dec is not None:
        dec = Decimal(dec[0]['value'])
    pixel_scale = header.get('SCALE')
    if pixel_scale is not None:
        pixel_scale = Decimal(pixel_scale[0]['value'])
    else:
        focal_len = header.get('FOCALLEN')
        pixel_size = header.get('XPIXSZ')
        if focal_len is not None and pixel_size is not None:
            focal_len = Decimal(focal_len[0]['value'])
            pixel_size = Decimal(pixel_size[0]['value'])
            pixel_scale = (pixel_size / focal_len) * Decimal(206.265)
    # Create SolveData object with the extracted RA, DEC and pixel scale
    plate_solve_data = SolveData(SkyCoord(
        ra, dec, unit=["deg", "deg"]), pixel_scale) if ra and dec and pixel_scale else None
    # Extract the latitude from the FITS header, default to 0 if not present
    lat = img_meta["FITSKeywords"].get("SITELAT")
    lat = Decimal(lat[0]['value'] if lat is not None else 0)
    # Extract the longitude from the FITS header, default to 0 if not present
    long = img_meta["FITSKeywords"].get("SITELONG")
    long = Decimal(long[0]['value'] if long is not None else 0)
    # Create SiteLocation object with the extracted latitude and longitude
    site_location = SiteLocation(lat=lat, long=long)
    # Create Header object with the extracted information
    header = Header(filename, exposure, timestamp, site_location, plate_solve_data)
    return header


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
                to_debayer: bool = False) -> None:
    try:
        # shm = SharedMemory(name=shm_params.shm_name, create=False)
        # imgs = np.ndarray(shape=shm_params.shm_shape, dtype=shm_params.shm_dtype, buffer=shm.buf)
        imgs = np.memmap(shm_params.shm_name, dtype=PIXEL_TYPE, mode='r+', shape=shm_params.shm_shape)
        for img_idx in indexes:
            img_data = load_image(file_list[img_idx], to_debayer)
            imgs[img_idx] = img_data
            progress_queue.put(img_idx)
            imgs.flush()
        # shm.close()
    except:
        import traceback
        traceback.print_exc()

def align_worker(img_indexes: list[int], shm_params: SharedMemoryParams, progress_queue: Queue) -> tuple[list[int], list[bool], list[np.ndarray]]:

    # shm = SharedMemory(name=shm_params.shm_name, create=False)
    # imgs = np.ndarray(shape=shm_params.shm_shape, dtype=shm_params.shm_dtype, buffer=shm.buf)
    imgs = np.memmap(shm_params.shm_name, dtype=PIXEL_TYPE, mode='r+', shape=shm_params.shm_shape)
    footprints = []
    successes = []
    if shm_params.y_slice.start is not None and shm_params.y_slice.stop is not None:
        target_y_slice = slice(
            shm_params.y_slice.start - SECONDARY_ALIGNMENT_OFFSET,
            shm_params.y_slice.stop + SECONDARY_ALIGNMENT_OFFSET)
    else:
        target_y_slice = shm_params.y_slice

    if shm_params.x_slice.start is not None and shm_params.x_slice.stop is not None:
        target_x_slice = slice(
            shm_params.x_slice.start - SECONDARY_ALIGNMENT_OFFSET,
            shm_params.x_slice.stop + SECONDARY_ALIGNMENT_OFFSET)
    else:
        target_x_slice = shm_params.x_slice
    for img_idx in img_indexes:
        try:
            imgs[img_idx, shm_params.y_slice, shm_params.x_slice], footprint = aa.register(
                imgs[img_idx, target_y_slice, target_x_slice],
                imgs[0, shm_params.y_slice, shm_params.x_slice],
                fill_value=0,
                max_control_points=50,
                min_area=5)
            imgs.flush()
        except Exception as e:
            footprint = None
            success = False
            # TODO log error with logger and print stacktrace
        else:
            success = True
        footprints.append(footprint)
        successes.append(success)

        progress_queue.put(img_idx)
    # shm.close()

    return img_indexes, successes, footprints


@measure_execution_time
def load_images(file_list: list[str], shm_params: SharedMemoryParams, to_debayer: bool = False,
                progress_bar: Optional[AbstractProgressBar] = None) -> None:
    available_cpus = cpu_count()
    used_cpus = min(available_cpus, len(file_list))
    with (Pool(processes=used_cpus) as pool):
        m = ctx.Manager()
        progress_queue = m.Queue()
        start = time.time()
        results = pool.map_async(
            partial(
                load_worker,
                file_list=file_list,
                shm_params=shm_params,
                to_debayer=to_debayer,
                progress_queue=progress_queue
            ),
            np.array_split(np.arange(len(file_list)), used_cpus))
        if progress_bar is not None:
            progress_bar.set_total(len(file_list))
            for _ in range(len(file_list)):
                img_idx = progress_queue.get()
                logger.log.debug(f"Loaded image '{file_list[img_idx]}' at index '{img_idx}'")
                progress_bar.update()
            progress_bar.complete()
        results.get()
        pool.close()
        pool.join()


@measure_execution_time
def align_images(shm_params: SharedMemoryParams,
                 progress_bar: Optional[AbstractProgressBar] = None) -> tuple[list[bool], np.ndarray]:

    available_cpus = cpu_count()
    frames_num = shm_params.shm_shape[0] - 1
    used_cpus = min(available_cpus, frames_num)
    with Pool(processes=used_cpus) as pool:
        m = ctx.Manager()
        progress_queue = m.Queue()
        results = pool.map_async(
            partial(align_worker, shm_params=shm_params, progress_queue=progress_queue),
            np.array_split(np.arange(1, frames_num + 1), used_cpus))

        if progress_bar is not None:
            progress_bar.set_total(frames_num)
            for _ in range(frames_num):
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
        footprint_map = [item[2] for item in res]
        pool.close()
        pool.join()
    return success_map, np.array(footprint_map)


@measure_execution_time
def stretch_images(shm_params: SharedMemoryParams,
                 progress_bar: Optional[AbstractProgressBar] = None) -> tuple[list[bool], np.ndarray]:

    available_cpus = cpu_count()
    frames_num = shm_params.shm_shape[0] - 1
    used_cpus = min(available_cpus, frames_num)
    with Pool(processes=used_cpus) as pool:
        m = ctx.Manager()
        progress_queue = m.Queue()
        results = pool.map_async(
            partial(stretch_worker, shm_params=shm_params, progress_queue=progress_queue),
            np.array_split(np.arange(frames_num + 1), used_cpus))

        if progress_bar is not None:
            progress_bar.set_total(frames_num)
            for _ in range(frames_num):
                img_idx = progress_queue.get()
                logger.log.debug(f"Stretched image at index '{img_idx}'")
                progress_bar.update()
            progress_bar.complete()
        results.get()


def stretch_worker(img_idexes: list[int], shm_params: SharedMemoryParams, progress_queue: ctx.Queue,
                   ) -> None:
    # shm = SharedMemory(name=shm_params.shm_name, create=False)
    # imgs = np.ndarray(shape=shm_params.shm_shape, dtype=shm_params.shm_dtype, buffer=shm.buf)
    try:
        imgs = np.memmap(shm_params.shm_name, dtype=shm_params.shm_dtype, mode='r+', shape=shm_params.shm_shape)
        for img_idx in img_idexes:
            img = imgs[img_idx, shm_params.y_slice, shm_params.x_slice]
            imgs[img_idx, shm_params.y_slice, shm_params.x_slice] = Stretch().stretch(img)
            progress_queue.put(img_idx)
            imgs.flush()
    except Exception as e:
        import traceback
        print(traceback.format_exc())

    # shm.close()


def plate_solve_image(image: np.ndarray, header: Header, sky_coord: Optional[np.ndarray] = None):
    # logger.log.info("Plate solving...")
    # and the size of its field of view
    header_data = header.solve_data
    pixel = header_data.pixel_scale * u.arcsec  # known pixel scale
    img = np.copy(image)
    img = np.reshape(img, (img.shape[0], img.shape[1]))
    # img = img * (256 * 256 - 1)
    # img = img.astype(np.uint16)

    shape = img.shape
    fov = np.min(shape[:2]) * pixel.to(u.deg)
    if sky_coord is None:
        sky_coord = twirl.gaia_radecs(header_data.sky_coord, fov)[0:200]
        sky_coord = twirl.geometry.sparsify(sky_coord, 0.1)
        sky_coord = sky_coord[:25]
    top_left_corner = (slice(None, img.shape[0] // 2), slice(None, img.shape[1] // 2), (0, 0))
    bottom_left_corner = (slice(img.shape[0] // 2, None), slice(None, img.shape[1] // 2), (img.shape[0]//2, 0))
    top_right_corner = (slice(None, img.shape[0] // 2), slice(img.shape[1] // 2, None), (0, img.shape[1]//2))
    bottom_right_corner = (slice(img.shape[0] // 2, None), slice(img.shape[1] // 2, None), (img.shape[0]//2, img.shape[1]//2))
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


    # detect stars in the image
    # tmp_pixel_coords = twirl.find_peaks(img, threshold=1)[0:200]
    # pixel_coords = []
    # for x, y in all_corner_peaks:
    #     dist_from_center_x = x - shape[1] // 2
    #     dist_from_center_y = y - shape[0] // 2
    #     if np.sqrt(dist_from_center_x ** 2 + dist_from_center_y ** 2) < min(shape) // 2:
    #         pixel_coords.append([x, y])
    # pixel_coords = np.array(pixel_coords[:25])
    # import matplotlib.pyplot as plt
    # plt.imshow(img, vmin=np.median(img), vmax=3 * np.median(img), cmap="Greys_r")
    # _ = CircularAperture(pixel_coords, r=10.0).plot(color="y")
    wcs = twirl.compute_wcs(all_corner_peaks, sky_coord, asterism=4)
    # logger.log.info(f"Image is plate solved successfully. Solution:\n{wcs}")
    return wcs, sky_coord

def plate_solve(shm_params: SharedMemoryParams, headers: list[Header],
                 progress_bar: Optional[AbstractProgressBar] = None):
    logger.log.info("Plate solving...")
    # shm = SharedMemory(name=shm_params.shm_name, create=False)
    # imgs = np.ndarray(shape=shm_params.shm_shape, dtype=shm_params.shm_dtype, buffer=shm.buf)
    imgs = np.memmap(shm_params.shm_name, dtype=shm_params.shm_dtype, mode='r+', shape=shm_params.shm_shape)
    _, reference_stars = plate_solve_image(imgs[0], headers[0])
    # shm.close()
    available_cpus = cpu_count()
    frames_num = shm_params.shm_shape[0]
    used_cpus = min(available_cpus, frames_num)
    with Pool(processes=used_cpus) as pool:
        m = ctx.Manager()
        progress_queue = m.Queue()
        results = pool.map_async(
            partial(plate_solve_worker, shm_params=shm_params, progress_queue=progress_queue, reference_stars=reference_stars, header=headers[0]),
            np.array_split(np.arange(frames_num), used_cpus))

        if progress_bar is not None:
            progress_bar.set_total(frames_num)
            for _ in range(frames_num):
                img_idx = progress_queue.get()
                logger.log.debug(f"Plate solved image at index '{img_idx}'")
                progress_bar.update()
            progress_bar.complete()
        res = results.get()
        new_res = []
        for item in res:
            new_res.extend(item)
        new_res.sort(key=lambda x: x[0])
    return [item[1] for item in new_res]

def plate_solve_worker(img_idexes: list[int], header: Header, shm_params: SharedMemoryParams,
                       reference_stars: np.ndarray, progress_queue: Queue):
    # shm = SharedMemory(name=shm_params.shm_name, create=False)
    # imgs = np.ndarray(shape=shm_params.shm_shape, dtype=shm_params.shm_dtype, buffer=shm.buf)
    imgs = np.memmap(shm_params.shm_name, dtype=shm_params.shm_dtype, mode='r', shape=shm_params.shm_shape)
    res = []
    for img_idx in img_idexes:
        img = imgs[img_idx]
        wcs, _ = plate_solve_image(img, header, reference_stars)
        progress_queue.put(img_idx)
        res.append((img_idx, wcs))
    # shm.close()
    return res

@measure_execution_time
def align_images_wcs(shm_params: SharedMemoryParams, all_wcses: list[WCS],
                     progress_bar: Optional[AbstractProgressBar] = None) -> tuple[list[bool], np.ndarray]:

    available_cpus = min(cpu_count(), 8)
    frames_num = shm_params.shm_shape[0]
    used_cpus = min(available_cpus, frames_num)
    with Pool(processes=used_cpus) as pool:
        m = ctx.Manager()
        progress_queue = m.Queue()
        results = pool.map_async(
            partial(align_wcs_worker, shm_params=shm_params, progress_queue=progress_queue, ref_wcs=all_wcses[0], all_wcses=all_wcses),
            np.array_split(np.arange(frames_num), used_cpus))

        if progress_bar is not None:
            progress_bar.set_total(frames_num)
            for _ in range(frames_num):
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
    return success_map, footprint_map
def align_wcs_worker(img_indexes: list[int], shm_params: SharedMemoryParams, progress_queue: Queue, ref_wcs: WCS, all_wcses: list[WCS]) -> tuple[list[int], list[bool], list[np.ndarray]]:

    # shm = SharedMemory(name=shm_params.shm_name, create=False)
    # imgs = np.ndarray(shape=shm_params.shm_shape, dtype=shm_params.shm_dtype, buffer=shm.buf)
    imgs = np.memmap(shm_params.shm_name, dtype=shm_params.shm_dtype, mode='r+', shape=shm_params.shm_shape)
    footprints = []
    successes = []
    for img_idx in img_indexes:
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
        except Exception as e:
            footprint = np.ones(shm_params.shm_shape[1:], dtype=bool)
            success = False
        else:
            success = True
            footprint = 1 - footprint
            footprint = np.array(footprint, dtype=bool)
        footprints.append(footprint)
        successes.append(success)

        progress_queue.put(img_idx)
    # shm.close()

    return img_indexes, successes, footprints

def make_file_list(folder: str) -> list[str]:
    import os
    file_list = os.listdir(folder)
    file_list = [os.path.join(folder, item) for item in file_list if item.lower().endswith(".xisf") or
                 item.lower().endswith(".fit") or item.lower().endswith(".fits")][:50]
    return file_list

# from progress_bar import ProgressBarCli
if __name__ == '__main__':
    pass
    # folder = "D:\\git\\dataset\\Virgo"
    # file_list = make_file_list(folder)
    # # source_data = SourceDataV2.load_headers(file_list)
    #
    # img = load_image(file_list[0])
    # shape = (len(file_list), *img.shape)
    # shm_name = 'bla'
    # shm = SharedMemory(name=shm_name, create=True, size=img.nbytes * len(file_list))
    # shm_params = SharedMemoryParams(
    #     shm_name=shm.name, shm_shape=shape, shm_size=img.nbytes * len(file_list), shm_dtype=img.dtype)
    # load_images(file_list, shm_params=shm_params, progress_bar=ProgressBarCli())

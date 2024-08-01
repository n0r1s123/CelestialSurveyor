import astropy.io.fits
import json
import numpy as np
import traceback

from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from datetime import datetime
from decimal import Decimal
from functools import partial
from logging.handlers import QueueHandler
from multiprocessing import Queue, cpu_count, Pool, Manager
from threading import Event
from typing import Optional
from xisf import XISF

from backend.progress_bar import AbstractProgressBar
from backend.data_classes import SolveData, Header
from logger.logger import get_logger
from backend.consuming_functions.measure_execution_time import measure_execution_time


logger = get_logger()


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


def load_headers_worker(filenames: list[str], progress_queue: Optional[Queue] = None,
                        stop_queue: Optional[Queue] = None, log_queue: Optional[Queue] = None) -> list[Header]:
    """
    Worker function to load header information from XISF and FIT(S) files.

    Args:
        filenames (List[str]): List of filenames to load headers from within this worker.
        progress_queue (Optional[Queue], optional): Queue for reporting progress. Defaults to None.
        stop_queue (Optional[Queue], optional): Queue to stop the loading process. Defaults to None.
        log_queue (Optional[Queue], optional): Queue for logging. Defaults to None.

    Returns:
        List[Header]: A list of Header objects containing the loaded header information from the files.
    """
    handler = QueueHandler(log_queue)
    logger.log.addHandler(handler)
    logger.log.debug(f"Load worker started with {len(filenames)} filenames")

    try:
        headers = []
        for filename in filenames:
            if stop_queue is not None and not stop_queue.empty():
                logger.log.debug("Load headers worker detected stop event. Stopping.")
                break
            if filename.lower().endswith(".xisf"):
                headers.append(load_header_xisf(filename))
            elif filename.lower().endswith(".fit") or filename.lower().endswith(".fits"):
                headers.append(load_header_fits(filename))
            else:
                raise ValueError("File type not supported. Supported file types: .xisf, .fit, .fits")
            if progress_queue is not None:
                progress_queue.put(True)
        return headers
    except Exception:
        logger.log.error(f"Load headers worker failed due to the following error:\n{traceback.format_exc()}")
        stop_queue.put("ERROR")
        raise


@measure_execution_time
def load_headers(filenames: list[str], progress_bar: Optional[AbstractProgressBar] = None,
                 stop_event: Optional[Event] = None) -> list[Header]:
    """
    Load header information from XISF and FIT(S) files using multiple workers.

    Args:
        filenames (List[str]): List of filenames to load headers from.
        progress_bar (Optional[AbstractProgressBar], optional): Progress bar for tracking loading progress.
            Defaults to None.
        stop_event (Optional[Event], optional): Event to stop the loading process. Defaults to None.

    Returns:
        List[Header]: A list of Header objects containing the loaded header information from the files.
    """
    available_cpu = min(4, cpu_count(), len(filenames))
    logger.log.debug(f"Number of CPUs to be used for alignment: {available_cpu}")
    with Pool(available_cpu) as pool:
        m = Manager()
        progress_queue = m.Queue()
        log_queue = m.Queue()
        logger.start_process_listener(log_queue)
        stop_queue = m.Queue(maxsize=1)
        logger.log.debug(f"Starting alignment with {available_cpu} workers")
        results = pool.map_async(
            partial(load_headers_worker, progress_queue=progress_queue, stop_queue=stop_queue, log_queue=log_queue),
            np.array_split(filenames, available_cpu))

        if progress_bar is not None:
            progress_bar.set_total(len(filenames))
            for _ in range(len(filenames)):
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
        headers = []
        for item in res:
            headers.extend(item)
        pool.close()
        pool.join()
        logger.log.debug(f"Load headers pool stopped.")
        logger.stop_process_listener()
    return headers


def load_header_xisf(filename: str) -> Header:
    """
    Load header information from an XISF file.

    Args:
        filename (str): The path to the XISF file.

    Returns:
        Header: A Header object containing the extracted information from the XISF file.
    """
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
    lat = lat[0]['value'] if lat is not None else 0
    # Extract the longitude from the FITS header, default to 0 if not present
    long = img_meta["FITSKeywords"].get("SITELONG")
    long = long[0]['value'] if long is not None else 0
    # Create SiteLocation object with the extracted latitude and longitude
    site_location = EarthLocation.from_geodetic(lat=lat, lon=long)
    # Create Header object with the extracted information
    header = Header(filename, exposure, timestamp, site_location, plate_solve_data)
    return header


def load_header_fits(filename: str) -> Header:
    """
    Load header information from a FITS file.

    Args:
        filename (str): The path to the FITS file.

    Returns:
        Header: A Header object containing the extracted information from the FITS file.
    """
    # Open the FITS file
    with astropy.io.fits.open(filename) as hdul:
        # Get the header of the first HDU
        header = hdul[0].header

        # Extract the exposure time from the header
        exposure = Decimal(header['EXPTIME'])

        # Extract the timestamp from the header and convert it to datetime
        timestamp = __get_datetime_from_str(header['DATE-OBS'])

        # Extract the right ascension (RA) from the header
        ra = header.get('RA') or header.get('OBJCTRA') or header.get('OBJRA')
        if not ra:
            logger.log.info(f"Header:\n{json.dumps(header, indent=4)}")
            raise ValueError("It's expected fits header to contain one of the following keywords: 'RA' or "
                             "'OBJCTRA'. None of these was found.")

        dec = header.get('DEC') or header.get('OBJCTDEC') or header.get('OBJDEC')
        if not dec:
            logger.log.info(f"Header:\n{json.dumps(header, indent=4)}")
            raise ValueError("It's expected fits header to contain one of the following keywords: 'DEC' or "
                             "'OBJCTDEC'. None of these was found.")

        # Extract the pixel scale from the header. If not present, calculate it from focal length and pixel size.
        pixel_scale = header.get('SCALE')
        if pixel_scale is not None:
            pixel_scale = abs(Decimal(pixel_scale))
        elif header.get('CDELT1'):
            pixel_scale = u.Quantity(header.get('CDELT1'), unit=header.get('CUNIT1', 'deg'))
            pixel_scale = abs(Decimal(pixel_scale.to(u.arcsec).value))
        else:
            focal_len = header.get('FOCALLEN')
            pixel_size = header.get('XPIXSZ')
            if focal_len is not None and pixel_size is not None:
                focal_len = Decimal(focal_len)
                pixel_size = Decimal(pixel_size)
                if focal_len != 0 and pixel_size != 0:
                    pixel_scale = (pixel_size / focal_len) * Decimal(206.265)
                else:
                    logger.log.info(f"Header:\n{header}")
                    raise ValueError("Pixel scale information is not present in FITS header")
            else:
                logger.log.info(f"Header:\n{header}")
                raise ValueError("Pixel scale information is not present in FITS header")
        # Create a SolveData object with the extracted RA, DEC and pixel scale
        plate_solve_data = SolveData(SkyCoord(ra, dec, unit=["deg", "deg"]), pixel_scale)

        # Extract the latitude and longitude from the header
        lat = header.get("SITELAT", 0)
        long = header.get("SITELONG", 0)

        # Create a SiteLocation object with the extracted latitude and longitude
        site_location = EarthLocation.from_geodetic(lat=lat, lon=long)

        # Create a Header object with the extracted information
        header = Header(filename, exposure, timestamp,  site_location, plate_solve_data)

        # Return the Header object
        return header


if __name__ == '__main__':
    a = EarthLocation.from_geodetic(
        lat='-44 01 01',
        lon='-52.67',
        height=0,
    )
    print(a.geodetic.lat.to_value())
    print(a.geodetic.lon.to_value())


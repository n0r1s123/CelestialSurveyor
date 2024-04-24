import astropy.io.fits
import cv2
import numpy as np
import os
import time
from backend.data_classes import SolveData, SiteLocation, Header
from datetime import datetime
from decimal import Decimal
from xisf import XISF
from logger.logger import get_logger
from multiprocessing import Process, Queue, cpu_count, Pool
from typing import Optional
from multiprocessing.shared_memory import SharedMemory
from functools import partial


logger = get_logger()

PIXEL_TYPE = "float32"


def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.log.debug(f"Execution time of {func.__name__}: {execution_time} seconds")
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
    return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f")


def load_header(filename: str, queue: Optional[Queue] = None) -> Header:
    """
    Load header information from a file based on its file extension.
    Args:
        filename (str): The name of the file to load the header from.
        queue (Optional[Queue]): The queue to send the header to.
    Returns:
        Header: The header information extracted from the file.
    Raises:
        ValueError: If the file type is not supported. Supported file types: .xisf, .fit, .fits
    """
    if filename.lower().endswith(".xisf"):
        header = load_header_xisf(filename)
    elif filename.lower().endswith(".fit") or filename.lower().endswith(".fits"):
        header = load_header_fits(filename)
    else:
        raise ValueError("File type not supported. Supported file types: .xisf, .fit, .fits")

    if queue is not None:
        queue.put(header)

    return header


def load_header_fits(filename: str) -> Header:
    """
    This function loads the header information from a FITS file.

    Args:
        filename (str): The path to the FITS file.

    Returns:
        Header: The header information extracted from the FITS file.

    Raises:
        ValueError: If the file type is not supported. Supported file types: .xisf, .fit, .fits
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
        plate_solve_data = SolveData(ra, dec, pixel_scale)

        # Extract the latitude and longitude from the header
        lat = Decimal(header.get("SITELAT", 0))
        long = Decimal(header.get("SITELONG", 0))

        # Create a SiteLocation object with the extracted latitude and longitude
        site_location = SiteLocation(lat=lat, long=long)

        # Create a Header object with the extracted information
        header = Header(filename, exposure, timestamp, plate_solve_data, site_location)

        # Return the Header object
        return header


def load_headers(filenames: list[str]) -> list[Header]:
    """
    This function loads headers from a list of filenames and returns a list of Header objects.

    Args:
        filenames (list[str]): List of filenames to load headers from.

    Returns:
        list[Header]: List of Header objects extracted from the filenames.
    """
    return [load_header(filename) for filename in filenames if filename.lower().endswith(".fits") or
            filename.lower().endswith(".xisf") or filename.lower().endswith(".fit")]


def load_header_xisf(filename: str) -> Header:
    """
    Load the header information from an XISF file.
    Args:
        filename (str): The path to the XISF file.
    Returns:
        Header: The header information extracted from the XISF file.
    Raises:
        KeyError: If the required keywords are not present in the FITS header.
    """
    # Initialize XISF object
    xisf = XISF(filename)
    # Get the metadata of the first image in the XISF file
    img_meta = xisf.get_images_metadata()[0]
    # Extract the timestamp from the FITS header
    timestamp = __get_datetime_from_str(img_meta["FITSKeywords"]["DATE-OBS"][0]['value'])
    # Extract the exposure time from the FITS header
    exposure = Decimal(img_meta["FITSKeywords"]["EXPTIME"][0]['value'])
    # Extract the right ascension (RA) from the FITS header
    ra = Decimal(img_meta["FITSKeywords"]["RA"][0]['value'])
    # Extract the declination (DEC) from the FITS header
    dec = Decimal(img_meta["FITSKeywords"]["DEC"][0]['value'])
    # Extract the pixel scale from the FITS header
    pixel_scale = Decimal(img_meta["FITSKeywords"]["SCALE"][0]['value'])
    # Create SolveData object with the extracted RA, DEC and pixel scale
    plate_solve_data = SolveData(ra, dec, pixel_scale)
    # Extract the latitude from the FITS header, default to 0 if not present
    lat = img_meta["FITSKeywords"].get("SITELAT")
    lat = Decimal(lat[0]['value'] if lat is not None else 0)
    # Extract the longitude from the FITS header, default to 0 if not present
    long = img_meta["FITSKeywords"].get("SITELONG")
    long = Decimal(long[0]['value'] if long is not None else 0)
    # Create SiteLocation object with the extracted latitude and longitude
    site_location = SiteLocation(lat=lat, long=long)
    # Create Header object with the extracted information
    header = Header(filename, exposure, timestamp, plate_solve_data, site_location)
    return header


def debayer(img_data):
    res = np.array(cv2.cvtColor(img_data, cv2.COLOR_BayerBG2GRAY), dtype=PIXEL_TYPE)
    res.reshape(img_data.shape[0], img_data.shape[1], 1)
    return res


def to_gray(img_data):
    return np.array(cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY), dtype=PIXEL_TYPE)


def load_image_fits(filename: str, to_debayer: bool = False) -> np.ndarray:
    """
    Load an image from a FITS file and convert it to a numpy array.
    Parameters:
        filename (str): The path to the FITS file.
        to_debayer (bool, optional): Whether to debayer the image if it is a single channel image.
            Defaults to False.
    Returns:
        np.ndarray: The loaded image as a numpy array. The shape of the array will be (height, width, channels).
            The channels will be 1 since all color channels are converted to grayscale.
    """
    with astropy.io.fits.open(filename) as hdul:
        img_data = np.array(hdul[0].data)
        if len(img_data.shape) == 2:
            img_data = img_data.reshape(img_data.shape[0], img_data.shape[1], 1)

        if img_data.shape[0] in [1, 3]:
            img_data = np.swapaxes(img_data, 0, 2)
        if to_debayer and img_data.shape[2] == 1:
            img_data = np.array(debayer(img_data), dtype=PIXEL_TYPE)
        img_data = img_data.reshape(img_data.shape[0], img_data.shape[1], 1)
        img_data = np.array(img_data, dtype=PIXEL_TYPE)
        if img_data.shape[2] == 3:
            img_data = np.array(to_gray(img_data), dtype=PIXEL_TYPE)
        # Normalize
        img_data /= 256 * 256 - 1
    return img_data


def load_image_xisf(filename: str, to_debayer: bool = False) -> np.ndarray:
    """
    Load an image from an XISF file and convert it to a numpy array.
    Parameters:
        filename (str): The path to the XISF file.
        to_debayer (bool, optional): is not used. Added for compatibility with other functions.
    Returns:
        np.ndarray: The loaded image as a numpy array. The shape of the array will be (height, width, channels).
            The channels will be 1 since all color channels are converted to grayscale.
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
        img_data = np.array(to_gray(img_data), dtype=PIXEL_TYPE)
    return img_data


def load_image(file_path: str, to_debayer: bool = False) -> np.ndarray:
    """
    Load an image from a file and convert it to a numpy array.
    Parameters:
        file_path (str): The path to the file.
        to_debayer (bool, optional): Whether to debayer the image if it is a single channel image.
            Defaults to False.
    Returns:
        np.ndarray: The loaded image as a numpy array. The shape of the array will be (height, width, channels).
            The channels will be 1 since all color channels are converted to grayscale.
    """
    if file_path.lower().endswith(".fits"):
        return load_image_fits(file_path, to_debayer)
    elif file_path.lower().endswith(".xisf"):
        return load_image_xisf(file_path, to_debayer)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def __load_images(__file_list: list[str], to_debayer: bool = False) -> np.ndarray:
    """
    Load images from a list of file paths and return them as a numpy array.

    Parameters:
        __file_list (list[str]): A list of file paths to load images from.
        to_debayer (bool, optional): Whether to debayer the images if they are single channel images. Defaults to False.

    Returns:
        np.ndarray: A numpy array containing the loaded images. The shape of the array will be (n, h, w, c), where n is the number of images, h and w are the height and width of the images, and c is the number of channels.
    """
    return np.array([load_image(file_path, to_debayer) for file_path in __file_list])


@measure_execution_time
def load_images(file_list: list[str], to_debayer: bool = False) -> np.ndarray:
    """
    Load images from a list of file paths in parallel and return them as a numpy array.

    Parameters:
        file_list (list[str]): A list of file paths to load images from.
        to_debayer (bool, optional): Whether to debayer the images if they are single channel images. Defaults to False.

    Returns:
        np.ndarray: A numpy array containing the loaded images.
    """
    file_list = [file_path for file_path in file_list if file_path.lower().endswith(".fits") or
                 file_path.lower().endswith(".xisf") or file_path.lower().endswith(".fit")]
    available_cpus = cpu_count()
    used_cpus = min(available_cpus, len(file_list))
    with Pool(processes=used_cpus) as pool:
        images = pool.map(partial(__load_images, to_debayer=to_debayer), np.array_split(file_list, used_cpus))

    images = np.concatenate(images)
    return np.array(images)


if __name__ == '__main__':
    folder = "D:\\git\\dataset\\Seahorse\\cropped"
    file_list = os.listdir(folder)
    file_list = [os.path.join(folder, item) for item in file_list]
    print(load_images(file_list, to_debayer=True).shape)

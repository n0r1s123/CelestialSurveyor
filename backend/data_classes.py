import numpy as np

from astropy.coordinates import SkyCoord, EarthLocation
from astropy.wcs import WCS
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional


@dataclass
class SolveData:
    """
    Class representing data required for plate solving of an image which is present in FITS or XISF header.

    Attributes:
        sky_coord (SkyCoord): The sky coordinates of image center.
        pixel_scale (Decimal): The pixel scale.
    """
    sky_coord: SkyCoord
    pixel_scale: Decimal

    def __post_init__(self):
        """
        Ensure the pixel scale is of Decimal type.
        """
        if not isinstance(self.pixel_scale, Decimal):
            raise TypeError("Pixel scale must be a Decimal object.")
        if not isinstance(self.sky_coord, SkyCoord):
            raise TypeError("Sky coordinates must be a SkyCoord object.")

    def __repr__(self):
        return f"SolveData(sky_coord={self.sky_coord}, pixel_scale={self.pixel_scale})"

    def __str__(self):
        return f"SolveData: Sky Coordinate: {self.sky_coord}, Pixel Scale: {self.pixel_scale}"


@dataclass
class Header:
    """
    Class to represent the header information of an image.

    Attributes:
        file_name (str): The name of the image file.
        exposure (Decimal): The exposure time of the image.
        timestamp (datetime): The timestamp of the image.
        site_location (EarthLocation): The geographical location of the site.
        solve_data (SolveData) [Optional]: The data required for plate solving of the image.
        wcs (WCS) [Optional]: WCS data - plate solve result.
    """
    file_name: str
    exposure: Decimal
    timestamp: datetime
    site_location: EarthLocation
    solve_data: Optional[SolveData] = None
    wcs: Optional[WCS] = None

    def __post_init__(self):
        """
        Perform type-checking on the attributes of the Header class to ensure correct data types.
        """
        if not isinstance(self.file_name, str):
            raise TypeError("File name must be a string.")

        if not isinstance(self.exposure, Decimal):
            raise TypeError("Exposure must be a Decimal object.")

        if not isinstance(self.timestamp, datetime):
            raise TypeError("Timestamp must be a datetime.datetime object.")

        if not isinstance(self.exposure, Decimal):
            raise TypeError("Exposure must be a Decimal object.")

        if not isinstance(self.site_location, EarthLocation):
            raise TypeError("Site location must be a EarthLocation object.")

        if not isinstance(self.solve_data, SolveData) and self.solve_data is not None:
            raise TypeError("Solve data must be a SolveData object or None.")

        if not isinstance(self.wcs, WCS) and self.wcs is not None:
            raise TypeError("WCS data must be a WCS object or None.")

    def __repr__(self):
        return (f"Header(file_name={self.file_name}, exposure={self.exposure}, timestamp={self.timestamp}, "
                f"site_location={self.site_location}, solve_data={self.solve_data}, wcs={self.wcs})")

    def __str__(self):
        return (f"Header: File Name: {self.file_name}, Exposure: {self.exposure}, Timestamp: {self.timestamp}, "
                f"Site Location: {self.site_location}, Solve Data: {self.solve_data}, WCS: {self.wcs}")


@dataclass
class SharedMemoryParams:
    """
    Data class representing parameters for shared memory file.
    Shared memory file is used to spread image data across multiple processes.

    Attributes:
        shm_name (str): The name of the shared memory.
        shm_size (int): The size of the shared memory.
        shm_shape (Tuple): The shape of the shared memory.
        shm_dtype (np.dtype): The data type of the shared memory.
        y_slice (slice): The y slice of the original image. Defaults to slice(None, None).
        x_slice (slice): The x slice of the original image. Defaults to slice(None, None).
    """
    shm_name: str
    shm_size: int
    shm_shape: tuple
    shm_dtype: np.dtype
    y_slice: slice = slice(None, None)
    x_slice: slice = slice(None, None)

    def __post_init__(self):
        if not isinstance(self.shm_name, str):
            raise TypeError("Shared memory name must be a string.")
        if not isinstance(self.shm_size, int):
            raise TypeError("Shared memory size must be an integer.")
        if not isinstance(self.shm_shape, tuple):
            raise TypeError("Shared memory shape must be a tuple.")
        if not isinstance(self.y_slice, slice):
            raise TypeError("Y slice must be a slice.")
        if not isinstance(self.x_slice, slice):
            raise TypeError("X slice must be a slice.")



    def __repr__(self):
        return (f"SharedMemoryParams(shm_name={self.shm_name}, shm_size={self.shm_size}, shm_shape={self.shm_shape}, "
                f"shm_dtype={self.shm_dtype}, y_slice={self.y_slice}, x_slice={self.x_slice})")

    def __str__(self):
        return (f"SharedMemoryParams: Shared Memory Name: {self.shm_name}, Shared Memory Size: {self.shm_size}, "
                f"Shared Memory Shape: {self.shm_shape}, Shared Memory Data Type: {self.shm_dtype}, "
                f"Y Slice: {self.y_slice}, X Slice: {self.x_slice}")

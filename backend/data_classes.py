import numpy as np
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime


@dataclass
class SolveData:
    """
    Class to hold astronomical solution data.

    Attributes:
        ra (Decimal): Right ascension in decimal degrees.
        dec (Decimal): Declination in decimal degrees.
        pixel_scale (Decimal): Scale of each pixel in arcseconds.
    """
    ra: Decimal
    dec: Decimal
    pixel_scale: Decimal

    def __post_init__(self):
        """
        Ensure the RA and DEC are of Decimal type.
        """
        if not isinstance(self.ra, Decimal):
            raise TypeError("RA must be a Decimal object.")
        if not isinstance(self.dec, Decimal):
            raise TypeError("DEC must be a Decimal object.")
        if not isinstance(self.pixel_scale, Decimal):
            raise TypeError("Pixel scale must be a Decimal object.")

    def __repr__(self):
        return f"SolveData(ra={self.ra}, dec={self.dec}, pixel_scale={self.pixel_scale})"

    def __str__(self):
        return f"SolveData: RA: {self.ra}, DEC: {self.dec}, Pixel Scale: {self.pixel_scale}"


@dataclass
class SiteLocation:
    """
    Class to represent a site's geographical location.

    Attributes:
        lat (Decimal): The latitude of the site in decimal degrees.
        long (Decimal): The longitude of the site in decimal degrees.
    """
    lat: Decimal
    long: Decimal

    def __post_init__(self):
        """
        Ensure the lat and long are of Decimal type.
        """
        if not isinstance(self.lat, Decimal):
            raise TypeError("Latitude must be a Decimal object.")
        if not isinstance(self.long, Decimal):
            raise TypeError("Longitude must be a Decimal object.")

    def __repr__(self):
        return f"SiteLocation(lat={self.lat}, long={self.long})"

    def __str__(self):
        return f"SiteLocation: Latitude: {self.lat}, Longitude: {self.long}"


@dataclass
class Header:
    """
    Represents header information.

    Attributes:
        file_name (str): The name of the FITS file.
        exposure (Decimal): The exposure value.
        timestamp (datetime): The timestamp of the header.
        solve_data (SolveData): The solve data associated with the header.
        site_location (SiteLocation): The site location associated with the header.
    """
    file_name: str
    exposure: Decimal
    timestamp: datetime
    solve_data: SolveData
    site_location: SiteLocation

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

        if not isinstance(self.solve_data, SolveData):
            raise TypeError("Solve data must be a SolveData object.")

        if not isinstance(self.exposure, Decimal):
            raise TypeError("Exposure must be a Decimal object.")

        if not isinstance(self.site_location, SiteLocation):
            raise TypeError("Site location must be a SiteLocation object.")

    def __repr__(self):
        return (f"Header(file_name={self.file_name}, exposure={self.exposure}, timestamp={self.timestamp}, "
                f"solve_data={self.solve_data}), site_location={self.site_location}")

    def __str__(self):
        return (f"Header: File Name: {self.file_name}, Exposure: {self.exposure}, Timestamp: {self.timestamp}, "
                f"Solve Data: {self.solve_data}, Site Location: {self.site_location}")


@dataclass
class Frame:
    """
    Represents a frame of an image with a header information.
    """
    image: np.ndarray
    header: Header

    def __post_init__(self):
        """
        Ensure the timestamp is of datetime.datetime type.
        """

        if not isinstance(self.image, np.ndarray):
            raise TypeError("Image must be a numpy array.")

        if not isinstance(self.header, Header):
            raise TypeError("Header must be a Header object.")

    def __repr__(self):
        return f"Frame(image={self.image}, header={self.header})"

    def __str__(self):
        return f"Frame: Image: {self.image}, Header: {self.header}"

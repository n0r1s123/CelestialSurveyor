from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS


class KnownObject:
    def __init__(self, properties: dict, wcs: WCS=None):
        self.name = properties["Object name"]
        self.coordinates = SkyCoord(
            ra=properties["Astrometric RA (hh:mm:ss)"].replace("'", "").replace('"', ''),
            dec=properties["Astrometric Dec (dd mm'ss\")"].replace("'", " ").replace('"', ''),
            unit=(u.hourangle, u.deg)
        )
        self.magnitude = float(properties["Visual magnitude (V)"])
        self._wcs = wcs
        self.__pixel_coordinates = None

    def __str__(self):
        return f"{self.name} {self.magnitude}\n{self.coordinates}"

    @property
    def wcs(self):
        return self._wcs

    @wcs.setter
    def wcs(self, value: WCS):
        self._wcs = value

    @property
    def pixel_coordinates(self):
        if not isinstance(self._wcs, WCS):
            raise ValueError("WCS is not specified or is not an instance of class WCS")
        return self.wcs.world_to_pixel(self.coordinates) if self.__pixel_coordinates is None else self.__pixel_coordinates


if __name__ == '__main__':
    params = {
        "Object name": "(2021 RP107)",
        "Astrometric RA (hh:mm:ss)": "03:14:38",
        "Astrometric Dec (dd mm'ss\")": "+31 41'05\"",
        "Dist. from center RA (\")": "-8.E3",
        "Dist. from center Dec (\")": "956.",
        "Dist. from center Norm (\")": "8268.",
        "Visual magnitude (V)": "19.9",
        "RA rate (\"/h)": "-4.002E+01",
        "Dec rate (\"/h)": "8.931E+00",
        "Est. error RA (\")": "2735.",
        "Est. error Dec (\")": "2735."
    }
    print(KnownObject(params))

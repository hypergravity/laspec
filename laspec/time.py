__all__ = ["datetime2jd", "jd2bjd"]

from typing import Union

from astropy import constants
from astropy import coordinates as coord
from astropy import units as u
from astropy.time import Time

SOL_kms = constants.c.to(u.km / u.s).value


def datetime2jd(
    datetime: str = "2018-10-24T05:07:06.0",
    format: str = "isot",
    tz_correction: float = 8,
):
    """
    Convert a datetime string to a Julian date.
    The format of the datetime string is assumed to be "isot" (ISO 8601).
    The timezone correction is subtracted from the Julian date to get the local time.

    Parameters
    ----------
    datetime : str
        The datetime string to be converted, default is "2018-10-24T05:07:06.0".
    format : str
        The format of the datetime string, default is "isot".
    tz_correction : int
        The timezone correction in hours, default is 8 (Beijing, GMT+8).

    Returns
    -------
    float
        The Julian date corresponding to the input datetime string
    """
    jd = Time(datetime, format=format).jd - tz_correction / 24.0
    return jd


def eval_ltt(
    ra=180.0,
    dec=30.0,
    jd=2456326.4583333,
    site=None,
    ephemeris="builtin",
):
    """
    Evaluate the light travel time for a given celestial object.

    Parameters
    ----------
    ra : float
        The right ascension of the celestial object in degrees, default is 180.0
    dec : float
        The declination of the celestial object in degrees, default is 30.0
    jd : float
        The Julian date, default is 2456326.4583333
    site : Union[None, coord.EarthLocation], optional
        The location of the observer on Earth, default is None, which uses the default location
        "Beijing Xinglong Observatory".
    ephemeris : str, optional
        The ephemeris used to calculate the light travel time, default is "builtin"

    Returns
    -------
    float
        The light travel time of the celestial object

    References
    ----------
    https://docs.astropy.org/en/stable/time/
    """
    if site is None:
        site = coord.EarthLocation.of_site("Beijing Xinglong Observatory")
    # sky position
    ip_peg = coord.SkyCoord(ra, dec, unit=(u.deg, u.deg), frame="icrs")
    # time
    times = Time(jd, format="jd", scale="utc", location=site)
    # evaluate ltt
    ltt_helio = times.light_travel_time(ip_peg, ephemeris=ephemeris)
    return ltt_helio.jd


def jd2bjd(
    ra: float = 180.0,
    dec: float = 30.0,
    jd: float = 2456326.4583333,
    site: Union[None, coord.EarthLocation] = None,
    ephemeris: str = "builtin",
) -> float:
    """
    Convert JD to BJD.

    Parameters
    ----------
    ra : float
        The right ascension of the celestial object in degrees, default is 180.0
    dec : float
        The declination of the celestial object in degrees, default is 30.0
    jd : float
        The Julian date, default is 2456326.4583333
    site : Union[None, coord.EarthLocation], optional
        The location of the observer on Earth, default is None, which uses the default location
        "Beijing Xinglong Observatory".
    ephemeris : str, optional
        The ephemeris used to calculate the light travel time, default is "builtin"

    Examples
    --------
    >>> from laspec.time import jd2bjd
    >>> jd2bjd(jd=2456326.4583333)
    2456326.463191622

    References
    ----------
    - https://docs.astropy.org/en/stable/time/
    - https://astroutils.astronomy.osu.edu/time/utc2bjd.html
    """
    # defaut site is Xinglong
    if site is None:
        site = coord.EarthLocation.of_site("Beijing Xinglong Observatory")
    # sky position
    ip_peg = coord.SkyCoord(ra, dec, unit=(u.deg, u.deg), frame="icrs")
    # time
    times = Time(jd, format="jd", scale="utc", location=site)
    # evaluate ltt
    ltt_bary = times.light_travel_time(ip_peg, ephemeris=ephemeris)
    time_barycentre = times.tdb + ltt_bary
    return time_barycentre.value

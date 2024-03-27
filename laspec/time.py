__all__ = ["datetime2jd", "eval_bjd"]

from typing import Union

from astropy import constants
from astropy import coordinates as coord
from astropy import units as u
from astropy.time import Time

SOL_kms = constants.c.value / 1000


def datetime2jd(datetime="2018-10-24T05:07:06.0", format="isot", tz_correction=8):
    jd = Time(datetime, format=format).jd - tz_correction / 24.0
    return jd


def eval_ltt(ra=180.0, dec=30.0, jd=2456326.4583333, site=None, ephemeris="builtin"):
    """evaluate the jd"""
    # conf: https://docs.astropy.org/en/stable/time/
    # defaut site is Xinglong
    if site is None:
        site = coord.EarthLocation.of_site("Beijing Xinglong Observatory")
    # sky position
    ip_peg = coord.SkyCoord(ra, dec, unit=(u.deg, u.deg), frame="icrs")
    # time
    times = Time(jd, format="jd", scale="utc", location=site)
    # evaluate ltt
    ltt_helio = times.light_travel_time(ip_peg, ephemeris=ephemeris)
    return ltt_helio.jd


def eval_bjd(
    ra: float = 180.0,
    dec: float = 30.0,
    jd: float = 2456326.4583333,
    site: Union[None, coord.EarthLocation] = None,
    ephemeris: str = "builtin",
) -> float:
    """Evaluate the bjd

    Examples
    --------
    >>> from laspec.time import eval_bjd
    >>> eval_bjd(jd=2456326.4583333)
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

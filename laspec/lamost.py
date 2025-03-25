import os
from collections.abc import Iterable
import numpy as np


def lamost_filepath(
    obsdate: str | Iterable[str],
    planid: str | Iterable[str],
    lmjd: int | Iterable[int],
    spid: int | Iterable[int],
    fiberid: int | Iterable[int],
    rootdir="",
    prefix="spec",
    extname="fits.gz",
):
    """generate file path of a LAMOST spectrum

    Parameters
    ----------
    obsdate: str
        obsdate, e.g. "2014-01-01"
    planid: str
        planid
    lmjd: int
        lmjd (use lmjd for DR3 and after!)
    spid: 2-digit integer
        spid, the number of the spectrogragh
    fiberid: int
        fiberid
    rootdir: str
        the root directory for storing spectra.
    prefix: str
        prefix of file name. "spec" for LRS and "med" for MRS.
    extname: str
        extension name, ".fits.gz" by default.

    Returns
    --------
    filepath: str or Iterable
        the path of root dir of directory (prefix).
        if un-specified, return file name.
    """
    if isinstance(lmjd, str):
        # scalar mode
        return f"{rootdir}/{obsdate.replace('-', '')}/{planid.strip()}/{prefix}-{lmjd:05d}-{planid.strip()}_sp{spid:02d}-{fiberid:03d}.{extname}"
    else:
        # vector mode
        fps = []
        for i in range(len(lmjd)):
            fps.append(
                f"{rootdir}/{obsdate[i].replace('-', '')}/{planid[i].strip()}/{prefix}-{lmjd[i]:05d}-{planid[i].strip()}_sp{spid[i]:02d}-{fiberid[i]:03d}.{extname}"
            )
        return np.array(fps)

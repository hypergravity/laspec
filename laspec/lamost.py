# -*- coding: utf-8 -*-
import os

import numpy as np


def lamost_filepath(planid, mjd, spid, fiberid, dirpath="", extname=".fits"):
    """ generate file path of a LAMOST spectrum

    Parameters
    ----------
    planid: string
        planid

    mjd: 5-digit integer
        mjd (use lmjd rather than mjd for DR3 and after!)

    spid: 2-digit integer
        spid, the number of the spectrogragh

    fiberid: 3-digit integer
        fiberid

    dirpath: string
        the root directory for storing spectra.

    Returns
    --------
    filepath: string
        the path of root dir of directory (prefix).
        if un-specified, return file name.

    """

    # pre-processing: strip
    if np.isscalar(planid):
        planid = planid.strip()
    else:
        planid = [_.strip() for _ in planid]

    if dirpath == "" or dirpath is None:
        # return file name
        if np.isscalar(mjd):
            # if only input one item
            return "spec-%05d-%s_sp%02d-%03d%s" \
                   % (mjd, planid, spid, fiberid, extname)
        else:
            # if input a list of items
            return np.array(["spec-%05d-%s_sp%02d-%03d%s" %
                             (mjd[i], planid[i], spid[i], fiberid[i], extname)
                             for i in range(len(mjd))])
    else:
        # return file path
        if not dirpath[-1] == os.path.sep:
            dirpath += os.path.sep

        if np.isscalar(mjd):
            # if only input one item
            return "%s%s%sspec-%05d-%s_sp%02d-%03d%s" \
                   % (dirpath, planid, os.path.sep,
                      mjd, planid, spid, fiberid, extname)
        else:
            # if input a list of items
            return np.array(["%s%s%sspec-%05d-%s_sp%02d-%03d%s" %
                             (dirpath, planid[i], os.path.sep, mjd[i],
                              planid[i], spid[i], fiberid[i], extname)
                             for i in range(len(mjd))])


def lamost_filepath_med(planid, mjd, spid, fiberid, dirpath="",
                        extname=".fits"):
    """ generate file path of a LAMOST spectrum (medium resolution)

    Parameters
    ----------
    planid: string
        planid

    mjd: 5-digit integer
        mjd (use lmjd rather than mjd for DR3 and after!)

    spid: 2-digit integer
        spid, the number of the spectrogragh

    fiberid: 3-digit integer
        fiberid

    dirpath: string
        the root directory for storing spectra.

    Returns
    --------
    filepath: string
        the path of root dir of directory (prefix).
        if un-specified, return file name.

    """

    # pre-processing: strip
    if np.isscalar(planid):
        planid = planid.strip()
    else:
        planid = [_.strip() for _ in planid]

    if dirpath == "" or dirpath is None:
        # return file name
        if np.isscalar(mjd):
            # if only input one item
            return "med-%05d-%s_sp%02d-%03d%s" \
                   % (mjd, planid, spid, fiberid, extname)
        else:
            # if input a list of items
            return np.array(["med-%05d-%s_sp%02d-%03d%s" %
                             (mjd[i], planid[i], spid[i], fiberid[i],
                              extname)
                             for i in range(len(mjd))])
    else:
        # return file path
        if not dirpath[-1] == os.path.sep:
            dirpath += os.path.sep

        if np.isscalar(mjd):
            # if only input one item
            return "%s%s%smed-%05d-%s_sp%02d-%03d%s" \
                   % (dirpath, planid, os.path.sep,
                      mjd, planid, spid, fiberid, extname)
        else:
            # if input a list of items
            return np.array(["%s%s%smed-%05d-%s_sp%02d-%03d%s" %
                             (dirpath, planid[i], os.path.sep, mjd[i],
                              planid[i], spid[i], fiberid[i], extname)
                             for i in range(len(mjd))])


def _test_lamost_filepath():
    """test function **lamost_filepath**
    """
    print(lamost_filepath("GAC_061N46_V3", 55939, 7, 78))
    print(lamost_filepath("GAC_061N46_V3", 55939, 7, 78, "/"))
    print(lamost_filepath("GAC_061N46_V3", 55939, 7, 78, "/pool"))
    print(lamost_filepath("GAC_061N46_V3", 55939, 7, 78, "/pool/"))


def sdss_filepath(plate, mjd, fiberid, dirpath="", extname=".fits"):
    """ generate file path of a LAMOST spectrum

    Parameters
    ----------
    plate: string
        plate

    mjd: 5-digit integer
        mjd (use lmjd rather than mjd for DR3 and after!)

    fiberid: 4-digit integer
        fiberid

    dirpath: string
        the root directory for storing spectra.

    extname: string
        in case that users want to synthesize other data format

    Returns
    --------
    filepath: string
        the path of root dir of directory (prefix).
        if un-specified, return file name.

    """

    if dirpath == "" or dirpath is None:
        # return file name
        if np.isscalar(mjd):
            # if only input one item
            return "spec-%04d-%05d-%04d%s" % (plate, mjd, fiberid, extname)
        else:
            # if input a list of items
            return np.array(["spec-%04d-%05d-%04d%s" %
                             (plate[i], mjd[i], fiberid[i], extname)
                             for i in range(len(mjd))])
    else:
        # return file path
        if not dirpath[-1] == os.path.sep:
            dirpath += os.path.sep

        if np.isscalar(mjd):
            # if only input one item
            return "%s%04d%sspec-%04d-%05d-%04d%s" \
                   % (dirpath, plate, os.path.sep,
                      plate, mjd, fiberid, extname)
        else:
            # if input a list of items
            return np.array(["%s%04d%sspec-%04d-%05d-%04d%s" %
                             (dirpath, plate[i], os.path.sep, plate[i],
                              mjd[i], fiberid[i], extname)
                             for i in range(len(mjd))])


def _test_sdss_filepath():
    print(sdss_filepath(2238, 52059, 1, "/"))


if __name__ == "__main__":
    print("")
    print("@Cham: start to test the module ...")
    print("")
    print("@Cham: testing ""lamost_filepath"" ...")
    _test_lamost_filepath()
    _test_sdss_filepath()
    print("@Cham: OK")

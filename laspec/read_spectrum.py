# -*- coding: utf-8 -*-
import os
from collections import OrderedDict

import numpy as np
from astropy.io import fits
from astropy.table import Table, Column

from .lamost import lamost_filepath
from .spec import Spec


def reconstruct_wcs_coord_from_fits_header(hdr, dim=1):
    """ reconstruct wcs coordinates (e.g., wavelength array) """
    # assert dim is not larger than limit
    assert dim <= hdr['NAXIS']

    # get keywords
    crval = hdr['CRVAL%d' % dim]
    cdelt = hdr['CDELT%d' % dim]
    try:
        crpix = hdr['CRPIX%d' % dim]
    except KeyError:
        crpix = 1

    # length of the current dimension
    naxis_ = hdr['NAXIS%d' % dim]

    # reconstruct wcs coordinates
    coord = np.arange(1 - crpix, naxis_ + 1 - crpix) * cdelt + crval
    return coord


def read_spectrum_phoenix_r10000(fp):
    """ read spectrum from PHOENIX R10000 """
    hl = fits.open(fp)
    wave = np.e ** reconstruct_wcs_coord_from_fits_header(hl[0].header)
    flux = hl[0].data
    return Table(data=[wave, flux], names=['wave', 'flux'])


def read_spectrum_elodie_r42000(fp):
    """ read spectrum from ELODIE library (R42000) """
    # assert the file exists
    assert os.path.exists(fp)

    # read fits
    hl = fits.open(fp)

    # reconstruct wave array
    wave = reconstruct_wcs_coord_from_fits_header(hl[0].header, dim=1)
    # flux
    flux = hl[0].data
    # flux err
    flux_err = hl[2].data
    # flux ivar
    flux_ivar = 1 / flux_err ** 2.

    # reconstruct spec
    sp = Spec(data=[wave, flux, flux_ivar, flux_err],
              names=['wave', 'flux', 'flux_ivar', 'flux_err'])
    return sp


def read_spectrum(filepath, filesource='auto'):
    """read SDSS/LAMOST spectrum

    Parameters
    ----------

    filepath: string
        input file path

    filesource: string
        {'sdss_dr12' / 'lamost_dr2' / 'lamost_dr3'}

    Returns
    -------

    specdata: astropy.table.Table
        spectra as a table

    """
    # auto-identify the spectrum origination
    if filesource == 'auto':
        telescope = fits.open(filepath)[0].header['TELESCOP']
        if telescope == 'SDSS 2.5-M':
            return read_spectrum(filepath, filesource='sdss_dr12')
        if telescope == 'LAMOST':
            return read_spectrum(filepath, filesource='lamost_dr3')

    # SDSS DR7 spectrum
    if filesource == 'sdss_dr7':
        hdulist = fits.open(filepath)

        # 1. flux, flux_err, mask
        data = hdulist[0].data  # 5 rows
        flux = data[0][:]
        flux_err = data[2][:]
        mask = data[3][:]

        # 2. wave
        # http://iraf.net/irafdocs/specwcs.php
        # wi = CRVALi + CDi_i * (li - CRPIXi)
        CRVAL1 = hdulist[0].header['CRVAL1']
        CD1_1 = hdulist[0].header['CD1_1']
        CRPIX1 = hdulist[0].header['CRPIX1']
        Npix = len(flux)
        wavelog = CRVAL1 + (np.arange(Npix) + 1 - CRPIX1) * CD1_1
        wave = np.power(10, wavelog)

        spec = Table([wave, flux, flux_err, mask],
                     names=['wave', 'flux', 'flux_err', 'mask'])
        return spec

    # SDSS DR10/DR12 spectrum
    if filesource == 'sdss_dr10' or filesource == 'sdss_dr12':
        data = fits.open(filepath)
        specdata = Table(data[1].data)
        wave = Column(name='wave', data=np.power(10., specdata['loglam']))
        flux_err = Column(name='flux_err', data=(specdata['ivar']) ** -0.5)
        specdata.add_columns([wave, flux_err])
        return specdata

    # LAMOST DR2/DR3 spectrum
    if filesource == 'lamost_dr3' or filesource == 'lamost_dr2' or filesource == 'lamost_dr1':
        data = fits.open(filepath)
        specdata = Table(data[0].data.T)
        flux = Column(name='flux', data=specdata['col0'])
        ivar = Column(name='ivar', data=specdata['col1'])
        flux_err = Column(name='flux_err', data=(specdata['col1']) ** -0.5)
        wave = Column(name='wave', data=specdata['col2'])
        and_mask = Column(name='and_mask', data=specdata['col3'])
        or_mask = Column(name='or_mask', data=specdata['col4'])
        # for flux_err, convert inf to nan
        flux_err[np.isinf(flux_err.data)] = np.nan
        return Table([wave, flux, flux_err, ivar, and_mask, or_mask])

    return None


def _test_read_spectrum():
    fp = '/home/cham/PycharmProjects/bopy/bopy/data/test_spectra/lamost_dr3/'\
         + lamost_filepath('GAC_061N46_V3', 55939, 7, 78)
    print(fp)
    sp = read_spectrum(fp)
    sp.pprint()


class MedSpec(OrderedDict):
    """ for Median Resolution Spectrum """
    meta = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        s = "No.    Name     Dimensions\n"
        for i, (k,v) in enumerate(self.items()):
            s += "{}   {}   {}Rx{}C\n".format(i, k, len(v), len(v.colnames))
        return s

    @staticmethod
    def read(fp):
        return read_lamostms(fp)


def read_lamostms(fp):
    # read fits
    hl = fits.open(fp)

    # initialize MS
    ms = MedSpec()

    # set meta
    ms.meta = OrderedDict(hl[0].header)

    # set spec
    for i, data in enumerate(hl[1:]):
        ms[data.name] = Table(data=data.data, meta=OrderedDict(data.header))

    return ms


if __name__ == '__main__':
    print('')
    print('@Cham: testing ''read_spectrum'' ...')
    _test_read_spectrum()

"""

=======================================
filesource:   'lamost_dr2'
=======================================
documents of data structures (LAMOST and SDSS spectra)
http://dr2.lamost.org/doc/data-production-description#toc_3

 RowNumber 	Data                Type
 1           Flux                float
 2           Inverse Variance 	 float
 3           WaveLength          float
 4           Andmask             float
 5           Ormask              float


=======================================
filesource:   'sdss_dr7'
=======================================
http://classic.sdss.org/dr7/dm/flatFiles/spSpec.html


=======================================
filesource:   'sdss_dr10' / 'sdss_dr12'
=======================================
http://data.sdss3.org/datamodel/files/BOSS_SPECTRO_REDUX/RUN2D/spectra/PLATE4/spec.html

 HDU 0  : Header info from spPlate
 HDU 1  : Coadded spectrum from spPlate --> use this
 HDU 2  : Summary metadata copied from spAll
 HDU 3  : Line fitting metadata from spZline
 HDU 4+ : [Optional] Individual spCFrame spectra [B, R for each exposure]


HDU 0: Header keywords only

Copied from spPlate with the following additions/modifications:

   PLUG_RA, PLUG_DEC, THING_ID, FIBERID: added from spAll
   NEXP and EXPID*: modified to just include the frames which contributed to this fiber
   Removed keywords which apply only to single exposures

HDU 1 (extname COADD): Coadded Spectrum from spPlate

Binary table with columns:
 Required    Columns
 Col     Name        Type        Comment
 1       flux        float32 	coadded calibrated flux [10-17 ergs/s/cm2/Å]
 2       loglam      float32 	log10(wavelength [Å])
 3       ivar        float32 	inverse variance of flux
 4       and_mask 	int32       AND mask
 5       or_mask 	int32       OR mask
 6       wdisp       float32 	wavelength dispersion in pixel=dloglam units
 7       sky         float32 	subtracted sky flux [10-17 ergs/s/cm2/Å]
 8       model       float32 	pipeline best model fit used for classification and redshift

"""
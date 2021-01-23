# -*- coding: utf-8 -*-

import numpy as np
from astropy.table import Table, Column


class Spec(Table):

    def __init__(self, *args, **kwargs):
        super(Spec, self).__init__(*args, **kwargs)
        # print self.colnames
        # assert 'wave' in self.colnames
        # assert 'flux' in self.colnames
        # why???

    def norm_spec_pixel(self, norm_wave):
        sub_nearest_pixel = np.argsort(np.abs(self['wave']-norm_wave))[0]
        self['flux'] /= self['flux'][sub_nearest_pixel]
        return self

    def norm_spec_median(self):
        self['flux'] /= np.median(self['flux'])
        return self

    def extract_chunk_wave_interval(self, wave_intervals=None):
        """ return spec chunk in a given wavelength interval """
        if wave_intervals is None:
            return self
        else:
            spec_chunks = []
            # should use ind (not sub)
            for wave_interval in wave_intervals:
                ind_wave_interval = np.logical_and(
                    self['wave'] >= wave_interval[0],
                    self['wave'] <= wave_interval[1])
                spec_chunks.append(self[ind_wave_interval])
            return spec_chunks


# ############################# #
# spectrum quick initialization #
# ############################# #

def spec_quick_init(wave, flux):
    """

    Parameters
    ----------
    wave : numpy.ndarray
        wavelength array

    flux : numpy.ndarray
        flux array

    Returns
    -------
    spec_ : bopy.spec.Spec
        Spec, at least contains 'wave' and 'flux' columns.

    """
    spec_ = Spec(
        [Column(np.array(wave), 'wave'),
         Column(np.array(flux), 'flux')])
    return spec_


def _test_spec_quick_init():
    wave = np.arange(5000., 7000., 0.91)
    flux = np.sin(wave/1000)*1000 + 30000. + np.random.rand(len(wave))
    spec_qi = spec_quick_init(wave, flux)
    spec_qi.pprint()
    print('--------------------------------------')
    print('@Cham: _test_spec_quick_init() OK ... ')
    print('--------------------------------------')


# ###################################### #
# continuum normalization for a spectrum #
# ###################################### #

def wave2ranges(wave, wave_intervals=None):
    """ convert wavelength intervals to (pixel) ranges """
    if wave_intervals is None:
        return None
    else:
        wave_intervals = np.array(wave_intervals)

        # assert wave_intervals is a 2-column array
        assert wave_intervals.shape[1] == 2

        ranges = np.zeros_like(wave_intervals)
        for i in range(len(wave_intervals)):
            ranges[i, 0] = np.sum(wave < wave_intervals[i, 0])
            ranges[i, 1] = np.sum(wave < wave_intervals[i, 1])
        return ranges


# ################################## #
# other ways to normalize a spectrum #
# ################################## #

def norm_spec_pixel(spec, norm_wave):
    sub_nearest_pixel = np.argsort(np.abs(spec['wave']-norm_wave))[0]
    spec['flux'] /= spec['flux'][sub_nearest_pixel]
    return spec


def norm_spec_median(spec):
    spec['flux'] /= np.median(spec['flux'])
    return spec


def norm_spec_chunk_median(spec_chunks):
    for i in range(len(spec_chunks)):
        spec_chunks[i]['flux'] /= np.median(spec_chunks[i]['flux'])
    return spec_chunks


# ##################################################### #
# it is useful to break spectrum into chunks, sometimes #
# ##################################################### #

def break_spectrum_into_chunks(spec, ranges=None, amp=None):
    if ranges is not None:
        # let's break this spectrum into chunks

        # re-format ranges into numpy.array
        ranges = np.array(ranges)

        # assert amp is None or a list with the same length
        if amp is None:
            amp = np.ones(len(ranges))
        assert len(amp) == len(ranges)

        # break spectrum into chunks
        spec_chunks = []
        for i in range(len(ranges)):
            ind_chunk = np.logical_and(spec['wave'] >= ranges[i][0], spec['wave'] <= ranges[i][0])
            spec_chunk = spec[ind_chunk]
            spec_chunk['flux'] *= amp[i]
            spec_chunks.append(spec_chunk)

    else:
        # then this spectrum is continuous, return itself
        return spec


def break_spectra_into_chunks(spec_list, ranges=None, amp=None):
    return [break_spectrum_into_chunks(spec, ranges, amp) for spec in spec_list]


if __name__ == '__main__':
    _test_spec_quick_init()
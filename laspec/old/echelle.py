# -*- coding: utf-8 -*-
import numpy as np
import glob
import os
import re
from astropy.io import fits
from astropy.table import Table, Column


def read_echelle_spec_ascii(fp):
    """ read echelle spectrum from ascii file

    Parameters
    ----------
    fp: string
        filepath

    Returns
    _______
    spec: astropy.table.Table
        spectrum

    """
    # sp = Table.read(fp, format='ascii.fast_no_header')
    # sp.colnames = ['wave', 'flux']
    data = np.loadtxt(fp)
    return Table(data, names=['wave', 'flux'])


def read_echelle_spec_ascii_dir(dirpath, ext='.dat'):
    """ read echelle spectra from ascii file directory
    """
    filelist = glob.glob(dirpath + '*' + '.dat')
    return [read_echelle_spec_ascii(filepath) for filepath in filelist]


def read_echelle_spec_ascii_flist(filelist):
    """ read echelle spectra from ascii file list
    """
    return [read_echelle_spec_ascii(filepath) for filepath in filelist]


class EchelleSpec:
    filepre = ''
    objname = ''
    order = np.array([])
    spec = []
    wave_start = np.array([])
    wave_end = np.array([])
    wave_med = np.array([])

    def __init__(self,
                 dirpath,
                 order='from_file_name',
                 extname='.dat',
                 objname=''):
        """

        Parameters
        ----------
        dirpath: string
            directory path

        order: ('from_file_path' | else)

        objname:

        Returns
        -------
        EchelleSpec

        """

        filelist = glob.glob(dirpath + '/*' + extname)
        basenamelist = [os.path.basename(filepath) for filepath in filelist]
        strschlist = [re.search('_\d+%s' % extname, basename).group(0) for basename in basenamelist]
        decdiglist = [strsch.replace('_', '').replace(extname, '') for strsch in strschlist]

        self.filepre = basenamelist[0].replace(strschlist[0], '')
        self.spec = read_echelle_spec_ascii_flist(filelist)
        self.objname = objname

        # extract order
        if order == 'from_file_name':
            self.order = np.array([np.int16(decdig) for decdig in decdiglist])
        else:
            self.order = np.arange(len(filelist)) + 1

        # sort spec
        num_order = len(self.spec)
        self.wave_start = np.zeros(num_order)
        self.wave_end = np.zeros(num_order)
        self.wave_med = np.zeros(num_order)
        for i in range(num_order):
            self.spec[i].sort('wave')
            self.wave_start[i], self.wave_end[i] = \
                self.spec[i]['wave'].data[0], self.spec[i]['wave'].data[-1]
            self.wave_med[i] = (self.wave_start[i] + self.wave_end[i]) / 2.

    def extract_order(self, order):
        if np.isscalar(order):
            sub = np.arange(len(self.order))[self.order == order]
            return self.spec[sub]
        else:
            return [self.extract_order(order_) for order_ in order]

    def extract_order_wave_nearest(self, wave):
        ind = np.argsort(np.abs(self.wave_med-wave))[0]
        return self.spec[ind]

    def extract_order_wave_allincluding(self, wave):
        ind = np.logical_and(self.wave_start-wave < 0.,
                             self.wave_end-wave > 0.)
        sub = np.arange(len(ind))[ind]
        return [self.spec[sub_] for sub_ in sub]


def test1():
    dirpath = '/home/cham/PycharmProjects/bopy/bopy/data/test_echelle/w20160119027s'
    filelist = glob.glob(dirpath + '/*.dat')
    print(len(filelist))
    print('------------')
    for f in filelist:
        print(f)
    print('------------')
    print(filelist[0])
    sp = read_echelle_spec_ascii(filelist[0])
    print(sp)


def test2():
    dirpath = '/home/cham/PycharmProjects/bopy/bopy/data/test_echelle/w20160119027s'
    es = EchelleSpec(dirpath)
    sp = es.extract_order_wave_nearest(5780)
    sp.pprint()
    print('----')
    sps = es.extract_order_wave_allincluding(5780)
    for sp_ in sps:
        sp_.pprint()


if __name__ == '__main__':
    test2()
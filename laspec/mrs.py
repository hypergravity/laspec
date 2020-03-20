import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy.io import fits
from astropy.table import Table

from .normalization import normalize_spectrum_iter


class MrsSpec:
    """ MRS spectrum """
    # original quantities
    wave = np.array([], dtype=np.float)
    flux = np.array([], dtype=np.float)
    ivar = np.array([], dtype=np.float)
    mask = np.array([], dtype=np.bool)  # True for problematic

    # normalized quantities
    flux_norm = np.array([], dtype=np.float)
    flux_cont = np.array([], dtype=np.float)
    ivar_norm = np.array([], dtype=np.float)

    # other information (optional)
    name = ""
    snr = 0
    exptime = 0
    lmjm = 0
    lmjmlist = ""
    lamplist = None
    info = {}
    rv = 0.

    # status
    isempty = True
    isnormalized = False

    # default settings for normalize_spectrum_iter
    norm_kwargs = dict(p=1e-6, q=0.5, binwidth=100., lu=(-2, 3), niter=3)

    def __init__(self, wave=None, flux=None, ivar=None, mask=None, info={}, normalize=False, **norm_kwargs):
        """ a general form of spectrum
        Parameters
        ----------
        wave:
            array, wavelength
        flux:
            array, flux
        ivar:
            array, ivar
        mask:
            array(int), 1 for problematic
        info:
            dict, information of this spectrum
        normalize:
            bool, if True, normalize spectrum after initialization
        norm_kwargs:
            normalization settings passed to normalize_spectrum_iter()
        """
        # set data
        if wave is not None and flux is not None:
            self.wave, self.flux = wave, flux
            self.isempty = False
        else:
            # a null spec
            self.wave = np.array([], dtype=np.float)
            self.flux = np.array([], dtype=np.float)
            self.isempty = True
        # ivar and mask is optional for spec
        if ivar is None:
            self.ivar = np.ones_like(self.flux, dtype=np.float)
        else:
            self.ivar = ivar
        if mask is None:
            self.mask = np.zeros_like(self.flux, dtype=np.bool)
        else:
            self.mask = mask
        # set info
        for k, v in info.items():
            self.__setattr__(k, v)
        self.info = info
        # update norm kwargs
        self.norm_kwargs.update(norm_kwargs)
        # normalize spectrum
        if normalize:
            self.normalize()
            self.isnormalized = True
        return

    def __repr__(self):
        return "<MrsSpec name={} snr={:.1f}>".format(self.name, self.snr)

    @staticmethod
    def from_hdu(hdu=None, normalize=True, **norm_kwargs):
        """ convert MRS HDU to spec """
        if hdu is None or hdu.header["EXTNAME"] == "Information":
            return MrsSpec()
        else:
            spec = Table(hdu.data)
            spec.sort("LOGLAM")
            if "COADD" in hdu.name:
                # it's coadded spec
                wave = 10 ** spec["LOGLAM"].data
                flux = spec["FLUX"].data
                ivar = spec["IVAR"].data
                mask = spec["ORMASK"].data > 0  # use ormask for coadded spec
                info = dict(name=hdu.header["EXTNAME"],
                            lmjmlist=hdu.header["LMJMLIST"],
                            snr=np.int(hdu.header["SNR"]),
                            lamplist=hdu.header["LAMPLIST"])
            elif hdu.name.startswith("B-") or hdu.name.startswith("R-"):
                # it's epoch spec
                wave = 10 ** spec["LOGLAM"].data
                flux = spec["FLUX"].data
                ivar = spec["IVAR"].data
                mask = spec["PIXMASK"].data > 0  # use pixmask for epoch spec
                info = dict(name=hdu.header["EXTNAME"],
                            lmjm=np.int(hdu.header["LMJM"]),
                            exptime=np.int(hdu.header["EXPTIME"]),
                            snr=np.int(hdu.header["SNR"]),
                            lamplist=hdu.header["LAMPLIST"])
            else:
                raise ValueError("@MrsFits: error in reading epoch spec!")
            # initiate MrsSpec
            return MrsSpec(wave, flux, ivar, mask, info=info, normalize=normalize, **norm_kwargs)

    def normalize(self, **norm_kwargs):
        """ normalize spectrum with (optional) new settings """
        if not self.isempty:
            # for normal spec
            # update norm kwargs
            self.norm_kwargs.update(norm_kwargs)
            # normalize spectrum
            self.flux_norm, self.flux_cont = normalize_spectrum_iter(self.wave, self.flux, **self.norm_kwargs)
            self.ivar_norm = self.ivar * self.flux_cont ** 2
        else:
            # for empty spec
            # update norm kwargs
            self.norm_kwargs.update(norm_kwargs)
            # normalize spectrum
            self.flux_norm, self.flux_cont, self.ivar_norm = None, None, None
            return

    def wave_rv(self, rv=None):
        """ return RV-corrected wavelength array
        Parameter
        ---------
        rv:
            float, radial velocity [km/s]
        """
        if rv is None:
            rv = self.rv
        return self.wave / (1 + rv * 1000 / const.c.value)


class MrsEpoch:
    """ MRS epoch spcetrum """
    nspec = 0
    speclist = []
    specnames = []
    # the most important attributes
    epoch = -1
    snr = []
    rv = 0.

    wave = np.array([], dtype=np.float)
    flux = np.array([], dtype=np.float)
    ivar = np.array([], dtype=np.float)
    mask = np.array([], dtype=np.int)
    flux_norm = np.array([], dtype=np.float)
    ivar_norm = np.array([], dtype=np.float)
    flux_cont = np.array([], dtype=np.float)

    # default settings for normalize_spectrum_iter
    norm_kwargs = dict(p=1e-6, q=0.5, binwidth=100, lu=(-2, 3), niter=3)

    def __init__(self, speclist, specnames=("B", "R"), epoch=-1, normalize=False, **norm_kwargs):
        """ combine B & R to an epoch spectrum
        In this list form, it is compatible with even echelle spectra

        speclist:
            spectrum list
        specnames:
            the names of spectra, will be used as suffix
        epoch:
            the epoch of this epoch spectrum
        normalize:
            if True, normalize spectra in initialization
        norm_kwargs:
            the normalization settings
        """
        # set epoch
        self.epoch = epoch

        # update norm kwargs
        self.norm_kwargs.update(norm_kwargs)

        self.nspec = len(speclist)
        # default name is spec order
        if specnames is None or len(specnames) == 0:
            specnames = [i for i in range(self.nspec)]
        self.speclist = speclist
        self.specnames = specnames

        # store spectrum data
        self.snr = []
        for i_spec in range(self.nspec):
            # get info
            self.snr.append(self.speclist[i_spec].snr)
            # normalize if necessary
            if normalize and not self.speclist[i_spec].isnormalized:
                self.speclist[i_spec].normalize(**self.norm_kwargs)
            # store each spec
            self.__setattr__("wave_{}".format(specnames[i_spec]), self.speclist[i_spec].wave)
            self.__setattr__("flux_{}".format(specnames[i_spec]), self.speclist[i_spec].flux)
            self.__setattr__("ivar_{}".format(specnames[i_spec]), self.speclist[i_spec].ivar)
            self.__setattr__("mask_{}".format(specnames[i_spec]), self.speclist[i_spec].mask)
            self.__setattr__("flux_norm_{}".format(specnames[i_spec]), self.speclist[i_spec].flux_norm)
            self.__setattr__("ivar_norm_{}".format(specnames[i_spec]), self.speclist[i_spec].ivar_norm)
            self.__setattr__("flux_cont_{}".format(specnames[i_spec]), self.speclist[i_spec].flux_cont)

        # concatenate into one epoch spec *
        for i_spec in range(self.nspec):
            self.wave = np.append(self.wave, self.speclist[i_spec].wave)
            self.flux = np.append(self.flux, self.speclist[i_spec].flux)
            self.ivar = np.append(self.ivar, self.speclist[i_spec].ivar)
            self.mask = np.append(self.mask, self.speclist[i_spec].mask)
            self.flux_norm = np.append(self.flux_norm, self.speclist[i_spec].flux_norm)
            self.ivar_norm = np.append(self.ivar_norm, self.speclist[i_spec].ivar_norm)
            self.flux_cont = np.append(self.flux_cont, self.speclist[i_spec].flux_cont)
        return

    def __repr__(self):
        s = "[MrsEpoch epoch={} nspec={}]".format(self.epoch, self.nspec)
        for i in range(self.nspec):
            s += "\n{}".format(self.speclist[i])
        return s

    def normalize(self, **norm_kwargs):
        """ normalize each spectrum with (optional) new settings """
        # update norm kwargs
        self.norm_kwargs.update(norm_kwargs)

        # normalize each spectrum
        for i_spec in range(self.nspec):
            self.speclist[i_spec].normalize(**self.norm_kwargs)

        self.flux_norm = np.array([], dtype=np.float)
        self.ivar_norm = np.array([], dtype=np.float)
        self.flux_cont = np.array([], dtype=np.float)
        # concatenate into one epoch spec
        for i_spec in range(self.nspec):
            if not self.speclist[i_spec].isempty:
                # self.wave = np.append(self.wave, self.speclist[i_spec].wave)
                # self.flux = np.append(self.flux, self.speclist[i_spec].flux)
                # self.ivar = np.append(self.ivar, self.speclist[i_spec].ivar)
                # self.mask = np.append(self.mask, self.speclist[i_spec].mask)
                self.flux_norm = np.append(self.flux_norm, self.speclist[i_spec].flux_norm)
                self.ivar_norm = np.append(self.ivar_norm, self.speclist[i_spec].ivar_norm)
                self.flux_cont = np.append(self.flux_cont, self.speclist[i_spec].flux_cont)
        return

    def wave_rv(self, rv=None):
        """ return RV-corrected wavelength array
        Parameter
        ---------
        rv:
            float, radial velocity [km/s]
        """
        if rv is None:
            rv = self.rv
        return self.wave / (1 + rv * 1000 / const.c.value)


class MrsFits(fits.HDUList):
    nhdu = 0
    hdunames = []
    isB = []
    isR = []
    isCoadd = []
    isEpoch = []
    ulmjm = []

    def __init__(self, fp=None):
        """ set file path and read data """
        # check fits existence
        if fp is None:
            print("@MrsSpec: file path is not set!")
        elif not os.path.exists(fp):
            raise RuntimeError("@MrsSpec: file not found! ", fp)
        else:
            self.filepath = fp
        # read HDU list
        super().__init__(fits.open(fp))
        # get HDU names
        self.nhdu = len(self)
        self.hdunames = [hdu.name for hdu in self]
        self.ulmjm = []

        self.isB = np.zeros(self.nhdu, dtype=np.bool)
        self.isR = np.zeros(self.nhdu, dtype=np.bool)
        self.isEpoch = np.zeros(self.nhdu, dtype=np.bool)
        self.isCoadd = np.zeros(self.nhdu, dtype=np.bool)
        self.lmjm = np.zeros(self.nhdu, dtype=np.int)
        for i in range(self.nhdu):
            if self.hdunames[i].startswith("B-"):
                self.isB[i] = True
                self.isEpoch[i] = True
                self.lmjm[i] = np.int(self.hdunames[i][2:])
            elif self.hdunames[i].startswith("R-"):
                self.isR[i] = True
                self.isEpoch[i] = True
                self.lmjm[i] = np.int(self.hdunames[i][2:])
            elif self.hdunames[i] == "COADD_B":
                self.isB[i] = True
                self.isCoadd[i] = True
            elif self.hdunames[i] == "COADD_R":
                self.isR[i] = True
                self.isCoadd[i] = True
            elif not self.hdunames[i] == "Information":
                raise RuntimeError("@MrsSpec: error during processing HDU name")

    def __repr__(self):
        """ as self.info()
        Summarize the info of the HDUs in this `HDUList`.
        Note that this function prints its results to the console---it
        does not return a value.
        """
        if self._file is None:
            name = '(No file associated with this HDUList)'
        else:
            name = self._file.name
        results = [f'Filename: {name}',
                   'No.    Name      Ver    Type      Cards   Dimensions   Format']
        format = '{:3d}  {:10}  {:3} {:11}  {:5d}   {}   {}   {}'
        default = ('', '', '', 0, (), '', '')
        for idx, hdu in enumerate(self):
            summary = hdu._summary()
            if len(summary) < len(default):
                summary += default[len(summary):]
            summary = (idx,) + summary
            results.append(format.format(*summary))
        return "\n".join(results[1:])

    # or get one spec (specify a key)?
    def get_one_spec(self, lmjm="COADD", band="B"):
        if lmjm == "COADD":
            k = "COADD_{}".format(band)
        else:
            k = "{}-{}".format(band, lmjm)
        return MrsSpec.from_hdu(self[k])

    def get_one_epoch(self, lmjm=84420148, normalize=True, norm_kwargs={}):
        """ get one epoch spec from fits """
        try:
            if isinstance(lmjm, str):
                assert lmjm is "COADD"
            if isinstance(lmjm, np.int):
                assert lmjm in self.lmjm
        except AssertionError:
            raise AssertionError("@MrsFits: lmjm={} is not found in this file!".format(lmjm))

        if lmjm == "COADD":
            kB = "COADD_B"
            kR = "COADD_R"
        else:
            kB = "B-{}".format(lmjm)
            kR = "R-{}".format(lmjm)
        # read B & R band spec
        if kB in self.hdunames:
            msB = MrsSpec.from_hdu(self[kB], normalize=normalize, **norm_kwargs)
        else:
            msB = MrsSpec(normalize=normalize, **norm_kwargs)
        if kR in self.hdunames:
            msR = MrsSpec.from_hdu(self[kR], normalize=normalize, **norm_kwargs)
        else:
            msR = MrsSpec(normalize=normalize, **norm_kwargs)
        # return MrsSpec
        return MrsEpoch((msB, msR), epoch=lmjm)

    def get_all_epochs(self, normalize=True, norm_kwargs={}, including_coadd=False):
        # make keys
        if not including_coadd:
            all_keys = []
        else:
            all_keys = ["COADD", ]
        all_keys.extend(np.unique(self.lmjm[self.lmjm > 0]))
        # return epochs
        return [self.get_one_epoch(k, normalize=normalize, norm_kwargs=norm_kwargs) for k in all_keys]

    @property
    def ls_epoch(self):
        return np.unique(self.lmjm[self.lmjm > 0])

    @property
    def ls_snr(self):
        return np.unique(self.snr[self.lmjm > 0])

    @property
    def epoch(self):
        return self.lmjm

    @property
    def snr(self):
        _snr = np.zeros((self.nhdu,), dtype=np.float)
        for i in range(self.nhdu):
            if "SNR" in self[i].header.keys():
                _snr[i] = self[i].header["SNR"]
        return _snr


class MrsSource(np.ndarray):
    """ array of MrsEpoch instances, """
    mes = []  # MrsEpoch list #
    name = ""  # source name

    @property
    def snr(self):
        return np.array([_.snr for _ in self], dtype=np.float)

    @property
    def epoch(self):
        return np.array([_.epoch for _ in self], dtype=np.float)

    @property
    def nepoch(self):
        return len(self)

    @property
    def rv(self):
        return np.array([_.rv for _ in self], dtype=np.float)

    def __new__(cls, data, name="", normalize=True, norm_kwargs={}, **kwargs):
        # prepare
        data = np.array(data, dtype=MrsEpoch)
        # sort
        indsort = np.argsort([_.epoch for _ in data])
        data = data[indsort]
        # substantiate
        msrc = super(MrsSource, cls).__new__(cls, buffer=data, dtype=data.dtype, shape=data.shape, **kwargs)
        # normalize if necessary
        if normalize:
            msrc.normalize(**norm_kwargs)
        return msrc

    # def __repr__(self):
    #     s = "{{MrsSource nepoch={} name={}}}\n".format(self.nepoch, self.name)
    #     for i in range(self.nepoch):
    #         s += self.mes[i].__repr__().split("\n")[1]
    #         s += "\n"
    #     return s

    @staticmethod
    def read(fps, normalize=True, **norm_kwargs):
        mes = []
        for fp in fps:
            mf = MrsFits(fp)
            mes.extend(mf.get_all_epochs(normalize=normalize, norm_kwargs={}))
        return MrsSource(mes, normalize=normalize, norm_kwargs=norm_kwargs)

    def normalize(self, **norm_kwargs):
        # normalization
        for i in range(self.nepoch):
            self[i].normalize(**norm_kwargs)
        return


if __name__ == "__main__":
    fp = "/Users/cham/PycharmProjects/laspec/laspec/data/KIC8098300/DR7_medium/med-58625-TD192102N424113K01_sp12-076.fits.gz"
    fps = glob.glob("/Users/cham/PycharmProjects/laspec/laspec/data/KIC8098300/DR7_medium/*.fits.gz")
    # read fits
    mf = MrsFits(fp)

    # print info
    mf.info()
    print(mf)

    # print all lmjm
    print(mf.ls_epoch)

    # get MRS spec from MrsFits
    specCoaddB = MrsSpec.from_hdu(mf["COADD_B"], normalize=False)
    msB = MrsSpec.from_hdu(mf["B-84420148"], normalize=True)
    msR = MrsSpec.from_hdu(mf["R-84420148"], normalize=True)
    print(msB, msR)
    print(msB.snr, msR.snr)

    # combine B and R into an epoch spec
    me = MrsEpoch([msB, msR], specnames=["B", "R"])
    print(me)

    # a short way of doing this:
    me = mf.get_one_epoch(84420148)
    print(me)
    me = mf.get_one_epoch("COADD")
    print(me, me.snr)
    mes = mf.get_all_epochs(including_coadd=False)
    print(mes)

    msrc = MrsSource(mes)
    msrc1 = MrsSource.read(fps)

    fig = plt.figure()
    plt.plot(me.wave, me.flux_norm)

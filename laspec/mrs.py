import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
from .normalization import normalize_spectrum_iter


class MrsSpec:
    """ MRS spectrum """
    # original quantities
    wave = None
    flux = None
    ivar = None
    mask = None

    # normalized quantities
    flux_norm = None
    flux_cont = None
    ivar_norm = None

    isempty = True

    # default settings for normalize_spectrum_iter
    norm_kwargs = dict(p=1e-6, q=0.5, binwidth=100., lu=(-2, 3), niter=3)

    def __init__(self, wave=None, flux=None, ivar=None, mask=None, normalize=False, **norm_kwargs):
        """ a general form of spectrum """
        self.wave, self.flux, self.ivar, self.mask = wave, flux, ivar, mask
        if self.wave is None and self.flux is None and self.ivar is None and self.mask is None:
            self.isempty = True
        else:
            self.isempty = False
        # update norm kwargs
        self.norm_kwargs.update(norm_kwargs)
        # normalize spectrum
        if normalize:
            self.normalize()
        return

    @staticmethod
    def read_mrs(hdu=None, normalize=True, **norm_kwargs):
        """ convert MRS HDU to spec """
        if hdu is None:
            return MrsSpec()
        else:
            spec = Table(hdu.data)
            spec.sort("LOGLAM")
            if "COADD" in hdu.name:
                # it's coadded spec
                wave = 10 ** spec["LOGLAM"].data
                flux = spec["FLUX"].data
                ivar = spec["IVAR"].data
                mask = spec["ORMASK"].data  # use ormask for coadded spec
            elif hdu.name.startswith("B-") or hdu.name.startswith("R-"):
                # it's epoch spec
                wave = 10 ** spec["LOGLAM"].data
                flux = spec["FLUX"].data
                ivar = spec["IVAR"].data
                mask = spec["PIXMASK"].data  # use pixmask for epoch spec
            else:
                raise ValueError("@MrsFits: error in reading epoch spec!")
            # initiate MrsSpec
            return MrsSpec(wave, flux, ivar, mask, normalize=normalize, **norm_kwargs)

    def normalize(self, **norm_kwargs):
        """ normalize spectrum with (optional) new settings """
        if not self.isempty:
            # update norm kwargs
            self.norm_kwargs.update(norm_kwargs)
            # normalize spectrum
            self.flux_norm, self.flux_cont = normalize_spectrum_iter(self.wave, self.flux, **self.norm_kwargs)
            self.ivar_norm = self.ivar * self.flux_cont ** 2
            return


class MrsEpoch:
    """ MRS epoch spcetrum """
    nspec = 0
    speclist = []
    specnames = []

    wave = np.array([], dtype=np.float)
    flux = np.array([], dtype=np.float)
    ivar = np.array([], dtype=np.float)
    mask = np.array([], dtype=np.float)
    flux_norm = np.array([], dtype=np.float)
    ivar_norm = np.array([], dtype=np.float)
    flux_cont = np.array([], dtype=np.float)

    # default settings for normalize_spectrum_iter
    norm_kwargs = dict(p=1e-6, q=0.5, binwidth=100, lu=(-2, 3), niter=3)

    def __init__(self, speclist, specnames=("B", "R"), normalize=False, **norm_kwargs):
        """ combine B & R to an epoch spectrum
        In this list form, it is compatible with even echelle spectra

        speclist:
            spectrum list
        specnames:
            the names of spectra, will be used as suffix
        normalize:
            if True, normalize spectra in initialization
        norm_kwargs:
            the normalization settings
        """
        # update norm kwargs
        self.norm_kwargs.update(norm_kwargs)

        self.nspec = len(speclist)
        # default name is spec order
        if len(specnames) == 0 or specnames is None:
            specnames = [i for i in range(self.nspec)]
        self.speclist = speclist
        self.specnames = specnames

        # store spectrum data
        for i_spec in range(self.nspec):
            this_spec = speclist[i_spec]
            assert this_spec is not None
            self.__setattr__("wave_{}".format(specnames[i_spec]), speclist[i_spec].wave)
            self.__setattr__("flux_{}".format(specnames[i_spec]), speclist[i_spec].flux)
            self.__setattr__("ivar_{}".format(specnames[i_spec]), speclist[i_spec].ivar)
            self.__setattr__("mask_{}".format(specnames[i_spec]), speclist[i_spec].mask)
            self.__setattr__("flux_norm_{}".format(specnames[i_spec]), speclist[i_spec].flux_norm)
            self.__setattr__("ivar_norm_{}".format(specnames[i_spec]), speclist[i_spec].ivar_norm)
            self.__setattr__("flux_cont_{}".format(specnames[i_spec]), speclist[i_spec].flux_cont)

        # if normalize
        if normalize:
            self.normalize(**self.norm_kwargs)

        # concatenate into one epoch spec
        for i_spec in range(self.nspec):
            if not self.speclist[i_spec].isempty:
                self.wave = np.append(self.wave, self.speclist[i_spec].wave)
                self.flux = np.append(self.flux, self.speclist[i_spec].flux)
                self.ivar = np.append(self.ivar, self.speclist[i_spec].ivar)
                self.mask = np.append(self.mask, self.speclist[i_spec].mask)
                self.flux_norm = np.append(self.flux_norm, self.speclist[i_spec].flux_norm)
                self.ivar_norm = np.append(self.ivar_norm, self.speclist[i_spec].ivar_norm)
                self.flux_cont = np.append(self.flux_cont, self.speclist[i_spec].flux_cont)
        return

    def normalize(self, **norm_kwargs):
        """ normalize each spectrum with (optional) new settings """
        # update norm kwargs
        self.norm_kwargs.update(norm_kwargs)

        # normalize each spectrum
        for i_spec in range(self.nspec):
            self.speclist[i_spec].normalize(**self.norm_kwargs)

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

    # get one spec (specify a key)
    # get one epoch (specify a lmjm)
    # get all epochs (specify a file path)
    # get all

    def get_one_spec(self, lmjm="COADD", band="B"):
        if lmjm == "COADD":
            k = "COADD_".format(band)
        else:
            k = "{}-{}".format(band, lmjm)
        return MrsSpec(ms[k])

    def get_one_epoch(self, lmjm=84420148):
        """ get one epoch spec from fits """
        try:
            assert lmjm in self.lmjm or lmjm == "COADD"
        except AssertionError:
            raise AssertionError("@MrsFits: lmjm={} is not found in this file!".format(lmjm))

        if lmjm == "COADD":
            kB = "COADD_B"
            kR = "COADD_R"
        else:
            kB = "B-{}".format(lmjm)
            kR = "R-{}".format(lmjm)
        # read B & R band spec
        msB = MrsSpec.read_mrs(self[kB])
        msR = MrsSpec.read_mrs(self[kR])

        # return MrsSpec
        return MrsEpoch((msB, msR))

    @property
    def lslmjm(self):
        return np.unique(self.lmjm[self.lmjm > 0])


if __name__ == "__main__":
    fp = "/Users/cham/PycharmProjects/laspec/laspec/data/KIC8098300/DR7_medium/med-58625-TD192102N424113K01_sp12-076.fits.gz"
    # read fits
    ms = MrsFits(fp)
    # print info
    ms.info()
    # print all lmjm
    print(ms.lslmjm)

    # get MRS spec from MrsFits
    specB = MrsSpec.read_mrs(ms["B-84420148"])
    specR = MrsSpec.read_mrs(ms["R-84420148"], normalize=True)
    # combine B and R into an epoch spec
    me = MrsEpoch([specB, specR], specnames=["B", "R"])

    # a short way of doing this:
    es = ms.get_one_epoch(84420148)
    es = ms.get_one_epoch("COADD")

    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(es.wave, es.flux_norm)

import os
import numpy as np
from astropy.io import fits
from astropy.table import Table


class MrsSpec:
    wave = None
    flux = None
    ivar = None
    mask = None

    def __init__(self, hdu=None):
        """ convert an HDU to spec """
        if hdu is None:
            return
        else:
            spec = Table(hdu.data)
            spec.sort("LOGLAM")
            if "COADD" in hdu.name:
                # it's coadded spec
                self.wave = 10 ** spec["LOGLAM"].data
                self.flux = spec["FLUX"].data
                self.ivar = spec["IVAR"].data
                self.mask = spec["OR_MASK"].data  # use pixmask for epoch spec
            elif hdu.name.startswith("B-") or hdu.name.startswith("R-"):
                # it's epoch spec
                self.wave = 10 ** spec["LOGLAM"].data
                self.flux = spec["FLUX"].data
                self.ivar = spec["IVAR"].data
                self.mask = spec["PIXMASK"].data  # use ormask instead for coadded spec
            else:
                raise ValueError("@MrsFits: error in reading epoch spec!")


class MrsEpoch:
    wave = None
    flux = None
    ivar = None
    mask = None

    waveB = None
    fluxB = None
    ivarB = None
    maskB = None

    waveR = None
    fluxR = None
    ivarR = None
    maskR = None

    def __init__(self, msB=None, msR=None):
        """ combine B & R to an epoch spectrum """
        assert msB is not None or msR is not None
        assert msB.wave is not None or msR.wave is not None

        # B & R spec
        if msB is not None:
            self.waveB = msB.wave
            self.fluxB = msB.flux
            self.ivarB = msB.ivar
            self.maskB = msB.mask
        if msR is not None:
            self.waveR = msR.wave
            self.fluxR = msR.flux
            self.ivarR = msR.ivar
            self.maskR = msR.mask

        # Epoch spec
        if msB is None and msR is not None:  # only R
            self.wave = np.copy(self.waveR)
            self.flux = np.copy(self.fluxR)
            self.ivar = np.copy(self.ivarR)
            self.mask = np.copy(self.maskR)
        elif msB is not None and msR is None:  # only B
            self.wave = np.copy(self.waveB)
            self.flux = np.copy(self.fluxB)
            self.ivar = np.copy(self.ivarB)
            self.mask = np.copy(self.maskB)
        elif msB is not None and msR is not None:  # B and R
            self.wave = np.hstack((self.waveB, self.waveR))
            self.flux = np.hstack((self.fluxB, self.fluxR))
            self.ivar = np.hstack((self.ivarB, self.ivarR))
            self.mask = np.hstack((self.maskB, self.maskR))
        else:
            raise ValueError("@MrsSpec: both B and R spec is None!")


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
        msB = MrsSpec(self[kB])
        msR = MrsSpec(self[kR])

        # return MrsSpec
        return MrsEpoch(msB, msR)

    @property
    def lslmjm(self):
        return np.unique(self.lmjm[self.lmjm > 0])


if __name__ == "__main__":
    fp = "/Users/cham/PycharmProjects/laspec/laspec/data/KIC8098300/DR7_medium/med-58625-TD192102N424113K01_sp12-076.fits.gz"
    ms = MrsFits(fp)
    ms.info()
    print(ms.lslmjm)
    es = ms.get_one_epoch(84420148)
    # hdulist = fits.open(fp)
    # nhdu = len(hdulist)
    # ks = [_ for i in range(nhdu)]

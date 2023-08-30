"""
This module aims to integrate some useful kits to cope with LAMOST data.

"""
import os.path

import numpy as np
import joblib
from astropy import table
from copy import deepcopy

from .mrs import MrsFits, MrsSpec
from . import PACKAGE_PATH


PATH_M9WAVEBR = PACKAGE_PATH + "/data/lamost/m9waveBR.dump"
PATH_L9WAVEBR = PACKAGE_PATH + "/data/lamost/l9waveBR.dump"


class LrsKit:
    """ a set of LAMOST MRS processing functions """

    @staticmethod
    def load_wave():
        """ load MRS wavelength (BR)

        Notes
        -----
        >>> wave_B = np.arange(3900, 5800, 1.1)
        >>> wave_R = np.arange(6200, 8800, 1.5)
        """
        return joblib.load(PATH_L9WAVEBR)

    @staticmethod
    def read_whole_spec(this_fp, rv_B=0., rv_R=0., wave_BR=None):
        """ read one whole LAMOST LRS spectrum

        Parameters
        ----------
        this_fp:
            file path
        rv_B:
            rv of blue arm
        rv_R:
            rv of red arm
        wave_BR:
            wavelength grid for interpolation

        Returns
        -------
        flux_norm, flux_norm_err

        """

        if wave_BR is None:
            wave_BR = LrsKit.load_wave()
        wave_B = wave_BR[wave_BR < 6000.]
        wave_R = wave_BR[wave_BR > 6000.]

        if not os.path.exists(this_fp):
            raise FileNotFoundError("{} not found!")

        try:
            # read spectrum
            this_spec = MrsSpec.from_lrs(this_fp, norm_type="spline", niter=2)

            # measure RV
            this_flux_norm_B, this_flux_norm_err_B = this_spec.interp_then_norm(wave_B, rv=rv_B)
            this_flux_norm_R, this_flux_norm_err_R = this_spec.interp_then_norm(wave_R, rv=rv_R)

            return np.hstack((this_flux_norm_B, this_flux_norm_R)), \
                np.hstack((this_flux_norm_err_B, this_flux_norm_err_R))
        except Exception as ae:
            # raise ae
            return np.zeros(len(wave_BR), float) * np.nan, np.zeros(len(wave_BR), float) * np.nan

    @staticmethod
    def generate_filepath(lmjd, planid, spid, fiberid, obsdate=None, dirspec=None):
        """ generate spectral file path """
        if dirspec is None:
            return "spec-{}-{}_sp{:02d}-{:03d}.fits.gz".format(
                lmjd, planid, spid, fiberid)
        else:
            return "{}/{}/{}/spec-{}-{}_sp{:02d}-{:03d}.fits.gz".format(
                dirspec, obsdate.replace("-", ""), planid, lmjd, planid, spid, fiberid)


class MrsKit:
    """ a set of LAMOST MRS processing functions """
    @staticmethod
    def read_multi_spec(fp_list, lmjm_list, rvzp_B_list, rvzp_R_list, wave_interp=None):
        """  read multiple spectra, interpolate to a wavelength grid

        Parameters
        ----------
        fp_list:
            file path list
        lmjm_list:
            lmjm list
        rvzp_B_list:
            rvzp (blue arm) list
        rvzp_R_list:
            rvzp (red arm) list
        wave_interp:
            the target wavelength grid

        Returns
        -------
        flux_norm, flux_norm_err, mask

        """
        n_spec = len(lmjm_list)

        flux_obs_list = []
        flux_err_obs_list = []
        mask_list = []

        for i_spec in range(n_spec):
            print("reading spectrum [{}/{}] ...".format(i_spec, n_spec))
            this_fp = fp_list[i_spec]
            this_lmjm = lmjm_list[i_spec]
            this_rvzp_B = rvzp_B_list[i_spec]
            this_rvzp_R = rvzp_R_list[i_spec]
            # read this epoch
            this_flux, this_flux_err, this_mask = MrsKit.read_single_epoch(
                this_fp, this_lmjm, this_rvzp_B, this_rvzp_R,
                wave_interp=wave_interp if wave_interp is not None else MrsKit.load_wave())
            # append data
            flux_obs_list.append(this_flux)
            flux_err_obs_list.append(this_flux_err)
            mask_list.append(this_mask)
        # return a regular ndarray
        return np.array(flux_obs_list), np.array(flux_err_obs_list), np.array(mask_list)

    @staticmethod
    def read_single_epoch(this_fp, this_lmjm, this_rvzp_B=0., this_rvzp_R=0., wave_interp=None):
        """ read a single epoch """
        # read spectra and reduce
        mf = MrsFits(this_fp)
        msB = mf.get_one_spec(lmjm=this_lmjm, band="B")
        msrB = msB.reduce(npix_cushion=70, norm_type="spline", niter=2)
        msR = mf.get_one_spec(lmjm=this_lmjm, band="R")
        msrR = msR.reduce(npix_cushion=70, norm_type="spline", niter=2)
        maskB = (msrB.mask > 0) | (msrB.indcr > 0) | (msrB.flux_norm <= 0) | (msrB.flux_norm >= 3)
        maskR = (msrR.mask > 0) | (msrR.indcr > 0) | (msrR.flux_norm <= 0) | (msrR.flux_norm >= 3)
        # shift rvzp
        wave_B = msrB.wave_rv(rv=-this_rvzp_B)
        wave_R = msrR.wave_rv(rv=-this_rvzp_R)

        # use default wavelength grid
        if wave_interp is None:
            wave_interp = MrsKit.load_wave()

        # interpolate spectrum
        wave_BR = np.hstack((wave_B, wave_R))
        this_flux = np.interp(wave_interp, wave_BR, np.hstack((msrB.flux_norm, msrR.flux_norm)))
        this_flux_err = np.interp(wave_interp, wave_BR, np.hstack((msrB.flux_norm_err, msrR.flux_norm_err)))
        this_mask = np.interp(wave_interp, wave_BR, np.hstack((maskB, maskR))) > 0
        return this_flux, this_flux_err, this_mask

    @staticmethod
    def load_wave():
        """ load MRS wavelength (BR) """
        return joblib.load(PATH_M9WAVEBR)


class CodeKit:

    @staticmethod
    def ezscatter(a, chunksize=1000, n_jobs=None):
        """ scatter array a to several jobs """
        if isinstance(a, int):
            a = np.arange(a, dtype=int)
        n_el = len(a)
        if n_jobs is not None:
            chunksize = int(np.ceil(n_el / n_jobs))
        n_chunks = int(np.ceil(n_el / chunksize))
        a_scattered = [a[chunksize * i_chunk:np.min((chunksize * (i_chunk + 1), n_el))] for i_chunk in range(n_chunks)]
        return a_scattered


class PubKit:
    """ toolkit for publishing data

    Examples
    --------
    >>> from laspec.lamost_kits import PubKit
    >>> PubKit.compress_table(tbl, tbl_name="m9", reserved=("bjd", "ra", "dec"))
    >>> # then copy the printed code and execute
    """

    @staticmethod
    def auto_compress(col, eps=1e-3, reserved=False):
        """
        auto compress int and float type data

        Parameters
        ----------
        col : astropy.table.Column or astropy.table.MaskedColumn
            the target column
        eps : float
            the precision loss tolerance
        reserved : bool
            the reserved columns (without modification)

        Returns
        -------
        auto compressed column
        """

        if reserved:
            print("  reserved column : *{}* ".format(col.name))
            return col

        # auto-compress int and float type data
        if col.dtype.kind == "i":
            alm = col.dtype.alignment
            original_dtype = "i{}".format(alm)
            while alm > 2:
                this_dtype = "i{}".format(alm)
                next_dtype = "i{}".format(alm // 2)
                if not np.all(col.astype(next_dtype) == col):
                    break
                alm //= 2
        elif col.dtype.kind == "f":
            alm = col.dtype.alignment
            original_dtype = "f{}".format(alm)
            while alm > 2:
                this_dtype = "f{}".format(alm)
                next_dtype = "f{}".format(alm // 2)
                if np.max(col.astype(next_dtype) - col) > eps:
                    break
                alm //= 2
        else:
            return col

        ccol = col.astype(this_dtype)
        print("compressed column : *{}* from {} to {}".format(col.name, original_dtype, this_dtype))
        return ccol

    @staticmethod
    def modify_column(tbl, colname, name=None, description=None, remove_mask=False, fill_value=None,
                      remove_directly=True, eps=2e-3, reserved=False):
        """ modify column

        Parameters
        ----------
        tbl: astropy.table.Table
            table
        colname: str
            column name
        name: str
            target name, if None, keep unchanged
        description:
            description of the column
        remove_mask:
            if True, remove mask and fill values
        fill_value:
            if None, use default fill_value
        remove_directly:
            if True, remove mask directly and keep data unchanged
        eps: float
            the tolerance of precision
        reserved: bool
            if True, reserve column

        Returns
        -------
        replace columns in-place
        """

        col = tbl[colname]

        if name is None:
            # change name if necessary
            name = col.name

        if description is None:
            # change description if necessary
            description = col.description

        if remove_directly:
            # remove mask directly
            data = col.data.data
            mcol = table.Column(data, name=name, description=description)

        elif isinstance(col, table.column.MaskedColumn):
            # for masked column
            data = col.data.data
            mask = col.data.mask

            # change dtype if necessary
            if fill_value is None:
                fill_value = col.data.fill_value
            data[mask] = fill_value

            if remove_mask:
                # remove mask
                mcol = table.Column(data, name=name, description=description)
            else:
                # keep masked
                mcol = table.MaskedColumn(data, mask=mask, name=name, fill_value=fill_value, description=description)
        else:
            # for normal Column
            data = col.data
            mcol = table.Column(data, name=name, description=description)

        # auto compress
        mcol = PubKit.auto_compress(mcol, eps=eps, reserved=reserved)
        # replace the column
        tbl.replace_column(colname, mcol)
        return

    @staticmethod
    def compress_table(tbl, tbl_name="tbl", eps=2e-3, reserved=("bjd", "ra", "dec")):
        """ compress table

        Parameters
        ----------
        tbl: astropy.table.Table
            table object
        tbl_name: str
            table name
        eps: float
            precision tolerance
        reserved: tuple or list
            reserved column names
        """

        infolist = []
        for colname in tbl.colnames:
            infodict = dict()
            infodict["colname"] = colname
            infodict["reserved"] = any([_name in colname.lower() for _name in reserved])
            infodict["dtype"] = tbl[colname].dtype.str
            infodict["dtype_kind"] = tbl[colname].dtype.kind
            infodict["description"] = tbl[colname].description

            # masked
            ismasked = isinstance(tbl[colname], table.column.MaskedColumn)
            if ismasked:
                infodict["masked"] = ismasked
                infodict["n_masked"] = np.sum(tbl[colname].mask)
                infodict["fill_value"] = tbl[colname].fill_value
            else:
                infodict["masked"] = ismasked
                infodict["n_masked"] = 0
                infodict["fill_value"] = None

            infolist.append(infodict)
        tinfo = table.Table(infolist)
        print(tinfo)
        print()
        print(" >>> Please copy & paste the code below to your editor/console and execute.")
        print(" >>> Modify the parameters for columns when necessary")
        print()

        code = ""
        for i in range(len(tinfo)):
            code += "PubKit.modify_column({}, ".format(tbl_name)
            code += "colname=\"{}\", ".format(tinfo[i]["colname"])
            code += "name=\"{}\", ".format(tinfo[i]["colname"])
            code += "description=\"\", ".format()
            code += "eps={}, ".format(eps)
            this_kwargs = dict(
                remove_mask=False,
                fill_value=None if tinfo[i]["dtype_kind"] != "f" else np.nan,
                remove_directly=tinfo[i]["n_masked"] == 0,
                reserved=tinfo[i]["reserved"],
            )
            for k, v in this_kwargs.items():
                code += "{}={}, ".format(k, v)
            code += ")\n"
        print(code)

        return

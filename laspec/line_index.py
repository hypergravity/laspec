# -*- coding: utf-8 -*-
import collections
import os

import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import LinearModel, GaussianModel

from .read_spectrum import read_spectrum


def integrate_spectrum(wave, flux_norm, flux_norm_err=None, mask=None, nmc=50, wave_range=(6554, 6574), suffix="Ha"):
    if not suffix == "":
        suffix = "_{}".format(suffix)

    rew = dict()
    ind_range = np.where(np.logical_and(wave > wave_range[0], wave < wave_range[1]))[0]
    npix_range = len(ind_range)
    if wave[0] < wave_range[0] < wave_range[1] < wave[-1] and npix_range >= 3:
        # good data
        wave_diff = np.diff(wave[ind_range])
        wave_diff = np.hstack((wave_diff[0], (wave_diff[1:] + wave_diff[:-1]) / 2, wave_diff[-1]))
        rew["EW{}".format(suffix)] = np.sum((1 - flux_norm[ind_range]) * wave_diff)

        if flux_norm_err is not None:
            # evaluate pct
            noise = np.random.normal(flux_norm[ind_range], flux_norm_err[ind_range], (nmc, npix_range))
            rew["EW16{}".format(suffix)], rew["EW50{}".format(suffix)], rew["EW84{}".format(suffix)] = \
                np.percentile(np.sum((1 - noise) * wave_diff, axis=1), [16, 50, 84])
        else:
            rew["EW16{}".format(suffix)] = np.nan
            rew["EW50{}".format(suffix)] = np.nan
            rew["EW84{}".format(suffix)] = np.nan

        if mask is not None:
            rew["EWnbad{}".format(suffix)] = np.sum(mask[ind_range] > 0)
        else:
            rew["EWnbad{}".format(suffix)] = 0
        return rew
    else:
        # bad data
        rew["EW{}".format(suffix)] = np.nan
        rew["EW16{}".format(suffix)] = np.nan
        rew["EW50{}".format(suffix)] = np.nan
        rew["EW84{}".format(suffix)] = np.nan
        rew["EWnbad{}".format(suffix)] = 0
        return rew


# old code below ----
# should consider whether to maintain filepath arg
# since the plot function could be replaced using recover
def measure_line_index(wave,
                       flux,
                       flux_err=None,
                       mask=None,
                       z=None,
                       line_info=None,
                       num_refit=(100, None),
                       filepath=None,
                       return_type='dict',
                       verbose=False):
    """ Measure line index / EW and have it plotted

    Parameters
    ----------
    wave: array-like
        wavelength vector

    flux: array-like
        flux vector

    flux_err: array-like
        flux error vector (optional)

        If un-specified, auto-generate an np.ones array

    mask: array-like
        andmask or ormask (optional)

        If un-specified, auto-generate an np.ones array (evenly weighted)

    line_info: dict
        information about spectral line, eg:

        >>> line_info_dib5780 = {'line_center':         5780,
        >>>                     'line_range':          (5775, 5785),
        >>>                     'line_shoulder_left':  (5755, 5775),
        >>>                     'line_shoulder_right': (5805, 5825)}

    num_refit: non-negative integer
        number of refitting.

        If 0, no refit will be performed

        If positive, refits will be performed after adding normal random noise

    z: float
        redshift (only specify when z is large)

    filepath: string
        path of the diagnostic figure

        if None, do nothing, else print diagnostic figure

    return_type: string
        'dict' or 'array'

        if 'array', np.array(return dict.values())

    verbose: bool
        if True, print details

    Returns
    -------
    line_indx: dict
        A dictionary type result of line index.
        If any problem encountered, return the default result (filled with nan).

    """
    try:
        # 0. do some input check
        # 0.1> check line_info
        line_info_keys = line_info.keys()
        assert 'line_range' in line_info_keys
        assert 'line_shoulder_left' in line_info_keys
        assert 'line_shoulder_right' in line_info_keys
        # 0.2> check line range/shoulder in spectral range
        assert np.min(wave) <= line_info['line_shoulder_left'][0]
        assert np.max(wave) >= line_info['line_shoulder_right'][0]

        # 1. get line information
        # line_center = line_info['line_center']  # not used
        line_range = line_info['line_range']
        line_shoulder_left = line_info['line_shoulder_left']
        line_shoulder_right = line_info['line_shoulder_right']

        # 2. data preparation
        # 2.1> shift spectra to rest-frame
        wave = np.array(wave)
        flux = np.array(flux)
        if z is not None:
            wave /= 1. + z
        # 2.2> generate flux_err and mask if un-specified
        if flux_err == None:
            flux_err = np.ones(wave.shape)
        if mask == None:
            mask = np.zeros(wave.shape)
        mask_ = np.zeros(wave.shape)
        ind_mask = np.all([mask!=0],axis=0)
        mask_[ind_mask] = 1
        mask = mask_

        # 3. estimate the local continuum
        # 3.1> shoulder wavelength range
        ind_shoulder = np.any([
            np.all([wave > line_shoulder_left[0],
                    wave < line_shoulder_left[1]], axis=0),
            np.all([wave > line_shoulder_right[0],
                    wave < line_shoulder_right[1]], axis=0)], axis=0)
        wave_shoulder = wave[ind_shoulder]
        flux_shoulder = flux[ind_shoulder]

        # 3.2> integrated/fitted wavelength range
        ind_range = np.logical_and(wave > line_range[0], wave < line_range[1])
        wave_range = wave[ind_range]
        flux_range = flux[ind_range]
        # flux_err_range = flux_err[ind_range]  # not used
        mask_range = mask[ind_range]
        flux_err_shoulder = flux_err[ind_shoulder]
        # mask_shoulder = mask[ind_shoulder]    # not used

        # 4. linear model
        mod_linear = LinearModel(prefix='mod_linear_')
        par_linear = mod_linear.guess(flux_shoulder, x=wave_shoulder)
        # ############################################# #
        # to see the parameter names:                   #
        # model_linear.param_names                      #
        # {'linear_fun_intercept', 'linear_fun_slope'}  #
        # ############################################# #
        out_linear = mod_linear.fit(flux_shoulder,
                                    par_linear,
                                    x=wave_shoulder,
                                    method='leastsq')

        # 5. estimate continuum
        cont_shoulder = out_linear.best_fit
        noise_std = np.std(flux_shoulder / cont_shoulder)
        cont_range = mod_linear.eval(out_linear.params, x=wave_range)
        resi_range = 1 - flux_range / cont_range

        # 6.1 Integrated EW (
        # estimate EW_int
        wave_diff = np.diff(wave_range)
        wave_step = np.mean(np.vstack([np.hstack([wave_diff[0], wave_diff]),
                                       np.hstack([wave_diff, wave_diff[-1]])]),
                            axis=0)
        EW_int = np.dot(resi_range, wave_step)

        # estimate EW_int_err
        num_refit_ = num_refit[0]
        if num_refit_ is not None and num_refit_>0:
            EW_int_err = np.std(np.dot(
                (resi_range.reshape(1, -1).repeat(num_refit_, axis=0) +
                 np.random.randn(num_refit_, resi_range.size) * noise_std),
                wave_step))

        # 6.2 Gaussian model
        # estimate EW_fit
        mod_gauss = GaussianModel(prefix='mod_gauss_')
        par_gauss = mod_gauss.guess(resi_range, x=wave_range)
        out_gauss = mod_gauss.fit(resi_range, par_gauss, x=wave_range)
        line_indx = collections.OrderedDict([
            ('SN_local_flux_err',        np.median(flux_shoulder / flux_err_shoulder)),
            ('SN_local_flux_std',        1. / noise_std),
            ('num_bad_pixel',            np.sum(mask_range != 0)),
            ('EW_int',                   EW_int),
            ('EW_int_err',               EW_int_err),
            ('mod_linear_slope',         out_linear.params[mod_linear.prefix + 'slope'].value),
            ('mod_linear_slope_err',     out_linear.params[mod_linear.prefix + 'slope'].stderr),
            ('mod_linear_intercept',     out_linear.params[mod_linear.prefix + 'intercept'].value),
            ('mod_linear_intercept_err', out_linear.params[mod_linear.prefix + 'intercept'].stderr),
            ('mod_gauss_amplitude',      out_gauss.params[mod_gauss.prefix + 'amplitude'].value),
            ('mod_gauss_amplitude_err',  out_gauss.params[mod_gauss.prefix + 'amplitude'].stderr),
            ('mod_gauss_center',         out_gauss.params[mod_gauss.prefix + 'center'].value),
            ('mod_gauss_center_err',     out_gauss.params[mod_gauss.prefix + 'center'].stderr),
            ('mod_gauss_sigma',          out_gauss.params[mod_gauss.prefix + 'sigma'].value),
            ('mod_gauss_sigma_err',      out_gauss.params[mod_gauss.prefix + 'sigma'].stderr),
            ('mod_gauss_amplitude_std',  np.nan),
            ('mod_gauss_center_std',     np.nan),
            ('mod_gauss_sigma_std',      np.nan)])

        # estimate EW_fit_err
        num_refit_ = num_refit[1]
        if num_refit_ is not None and num_refit_ > 2:
            # {'mod_gauss_amplitude',
            #  'mod_gauss_center',
            #  'mod_gauss_fwhm',
            #  'mod_gauss_sigma'}
            out_gauss_refit_amplitude = np.zeros(num_refit_)
            out_gauss_refit_center = np.zeros(num_refit_)
            out_gauss_refit_sigma = np.zeros(num_refit_)
            # noise_fit = np.random.randn(num_refit,resi_range.size)*noise_std
            for i in range(int(num_refit_)):
                # resi_range_with_noise = resi_range + noise_fit[i,:]
                resi_range_with_noise = resi_range + \
                                        np.random.randn(resi_range.size) * noise_std
                out_gauss_refit = mod_gauss.fit(resi_range_with_noise,
                                                par_gauss,
                                                x=wave_range)
                out_gauss_refit_amplitude[i],\
                out_gauss_refit_center[i],\
                out_gauss_refit_sigma[i] =\
                    out_gauss_refit.params[mod_gauss.prefix + 'amplitude'].value,\
                    out_gauss_refit.params[mod_gauss.prefix + 'center'].value,\
                    out_gauss_refit.params[mod_gauss.prefix + 'sigma'].value
                print(out_gauss_refit_amplitude[i], out_gauss_refit_center[i], out_gauss_refit_sigma[i])
            line_indx.update([
                              ('mod_gauss_amplitude_std', np.nanstd(out_gauss_refit_amplitude)),
                              ('mod_gauss_center_std',    np.nanstd(out_gauss_refit_center)),
                              ('mod_gauss_sigma_std',     np.nanstd(out_gauss_refit_sigma))
                              ])

        # 7. plot and save image
        if filepath is not None and os.path.exists(os.path.dirname(filepath)):
            save_image_line_indice(filepath, wave, flux, ind_range, cont_range,
                                   ind_shoulder, line_info)

        # if necessary, convert to array
        # NOTE: for a non-ordered dict the order of keys and values may change!
        if return_type == 'array':
            return np.array(line_indx.values())
        return line_indx
    except Exception:
        return measure_line_index_null_result(return_type)


def measure_line_index_null_result(return_type):
    """generate default value (nan/False) when measurement fails
    Returns
    -------
    default value (nan/False)
    """
    line_indx = collections.OrderedDict([
                  ('SN_local_flux_err',        np.nan),
                  ('SN_local_flux_std',        np.nan),
                  ('num_bad_pixel',            np.nan),
                  ('EW_int',                   np.nan),
                  ('EW_int_err',               np.nan),
                  ('mod_linear_slope',         np.nan),
                  ('mod_linear_slope_err',     np.nan),
                  ('mod_linear_intercept',     np.nan),
                  ('mod_linear_intercept_err', np.nan),
                  ('mod_gauss_amplitude',      np.nan),
                  ('mod_gauss_amplitude_err',  np.nan),
                  ('mod_gauss_center',         np.nan),
                  ('mod_gauss_center_err',     np.nan),
                  ('mod_gauss_sigma',          np.nan),
                  ('mod_gauss_sigma_err',      np.nan),
                  ('mod_gauss_amplitude_std',  np.nan),
                  ('mod_gauss_center_std',     np.nan),
                  ('mod_gauss_sigma_std',      np.nan)])
    if return_type == 'array':
        return np.array(line_indx.values())
    return line_indx


# pure fit:    100 loops, best of 3: 8.06 ms per loop (1 int + 1 fit)
# 1000 re-fit: 1 loops, best of 3: 378 ms per loop (1 int + 1 fit + 100 re-fit)


def measure_line_index_loopfun(filepath):
    """loopfun for measuring line index

    Parameters
    ----------
    filepath: string
        path of the spec document
    
    Returns
    -------
    several line_indx: tuple
        every line_indx is a dictionary type result of line index.
    """
    num_refit = 100, None
    return_type = 'array'
    line_info_dib5780 = {'line_center':         5780,
                         'line_range':          (5775, 5785),
                         'line_shoulder_left':  (5755, 5775),
                         'line_shoulder_right': (5805, 5825)}
    line_info_dib5797 = {'line_center':         5797,
                         'line_range':          (5792, 5802),
                         'line_shoulder_left':  (5755, 5775),
                         'line_shoulder_right': (5805, 5825)}
    line_info_dib6284 = {'line_center':         6285,
                         'line_range':          (6280, 6290),
                         'line_shoulder_left':  (6260, 6280),
                         'line_shoulder_right': (6310, 6330)}
    try:
        # read spectrum
        # -------------
        spec = read_spectrum(filepath, 'auto')

        # measure DIBs
        # ------------
        # DIB5780
        line_indx_dib5780 = measure_line_index(wave=spec['wave'],
                                               flux=spec['flux'],
                                               flux_err=spec['flux_err'],
                                               mask=spec['and_mask'],
                                               line_info=line_info_dib5780,
                                               num_refit=num_refit,
                                               return_type=return_type,
                                               z=0)
        # DIB5797
        line_indx_dib5797 = measure_line_index(wave=spec['wave'],
                                               flux=spec['flux'],
                                               flux_err=spec['flux_err'],
                                               mask=spec['and_mask'],
                                               line_info=line_info_dib5797,
                                               num_refit=num_refit,
                                               return_type=return_type,
                                               z=0)
        # DIB6284 
        line_indx_dib6284 = measure_line_index(wave=spec['wave'],
                                               flux=spec['flux'],
                                               flux_err=spec['flux_err'],
                                               mask=spec['and_mask'],
                                               line_info=line_info_dib6284,
                                               num_refit=num_refit,
                                               return_type=return_type,
                                               z=0)
        return line_indx_dib5780, line_indx_dib5797, line_indx_dib6284
    except Exception:
        return (measure_line_index_null_result(return_type),
                measure_line_index_null_result(return_type),
                measure_line_index_null_result(return_type))


def measure_line_index_recover_spectrum(wave, params, norm=False):
    """ recover the fitted line profile from params

    Parameters
    ----------
    wave: array-like
        the wavelength to which the recovered flux correspond

    params: 5-element tuple
        the 1 to 5 elements are:
        mod_linear_slope
        mod_linear_intercept
        mod_gauss_amplitude
        mod_gauss_center
        mod_gauss_sigma

    norm: bool
        if True, linear model (continuum) is deprecated
        else linear + Gaussian model is used

    """
    from lmfit.models import LinearModel, GaussianModel
    mod_linear = LinearModel(prefix='mod_linear_')
    mod_gauss = GaussianModel(prefix='mod_gauss_')
    par_linear = mod_linear.make_params()
    par_gauss = mod_gauss.make_params()
    par_linear['mod_linear_slope'].value = params[0]
    par_linear['mod_linear_intercept'].value = params[1]
    par_gauss['mod_gauss_amplitude'].value = params[2]
    par_gauss['mod_gauss_center'].value = params[3]
    par_gauss['mod_gauss_sigma'].value = params[4]
    if not norm:
        flux = 1 - mod_gauss.eval(params=par_gauss, x=wave)
    else:
        flux = \
            (1 - mod_gauss.eval(params=par_gauss, x=wave)) * \
            mod_linear.eval(params=par_linear, x=wave)
    return flux


def save_image_line_indice(filepath, wave, flux, ind_range, cont_range,
                           ind_shoulder, line_info):
    """Plot a line indice and save it as a .png document.

    Parameters
    ----------
    filepath: string
        path of the spec document

    wave: array
        wavelength vector

    flux: array
        flux vector

    ind_range: array
        bool indicating the middle range of a particular line

    cont_range: array
        continuum flux of the middle range derived from linear model

    ind_shoulder: array
        bool indicating the shoulder range of a particular line

    line_info: dict
        information about spectral line, eg:

        >>> line_info_dib5780 = {'line_center':         5780,
        >>>                      'line_range':          (5775, 5785),
        >>>                      'line_shoulder_left':  (5755, 5775),
        >>>                      'line_shoulder_right': (5805, 5825)}

    """
    filename = os.path.basename(filepath)
    fig = plt.figure()
    plt.plot(wave[ind_range], flux[ind_range], 'r-')
    plt.plot(wave[ind_range], cont_range, 'b-')
    plt.plot(wave[ind_shoulder], flux[ind_shoulder], 'm-')
    plt.title(r'line' + str(line_info['line_center']) + r'of ' + filename)
    fig.savefig(filepath)


def test_():
    # filepath = walk_dir()
    # filesource = 'auto'
    filepath = r'/pool/lamost/dr2/spectra/fits/F5902/spec-55859-F5902_sp01-001.fits'
    filesource = 'lamost_dr2'
    spec = read_spectrum(filepath, filesource)
    # 10 loops, best of 3: 35.7 ms per loop
    # line_indx_pack = measure_line_index_loopfun(filepath)
    z = 0.00205785
    line_info_dib6284 = {'line_center':         6285,
                         'line_range':          (6280, 6290),
                         'line_shoulder_left':  (6260, 6280),
                         'line_shoulder_right': (6310, 6330)}
    line_indx = measure_line_index(wave=spec['wave'],
                                   flux=spec['flux'],
                                   flux_err=spec['flux_err'],
                                   mask=spec['and_mask'],
                                   line_info=line_info_dib6284,
                                   num_refit=(100, 0),
                                   return_type='dict',
                                   z=z)
    for key in line_indx.keys():
        print (key, line_indx[key])
    print(np.sum(np.isnan(line_indx.values())))

    '''
    45 ms for integration and other procedures
    380 ms for 100 refits
    In the fastest way (45ms), run 40 line indices on 4 million spectra:
    0.045*40*4E6/24/86400 ~ 3.5 days
    In the slowest way (380ms)
    0.420*40*4E6/24/86400 ~ 32.5 days
    '''


# I don't think this function should be implemented here,
# it could be useful if under the bopy.core package
def walk_dir(dirpath):
    """ enumerate all files under dirpath

    Parameters
    ----------
    dirpath: string
        the directory to be walked in

    Returns
    -------
    filename: list
        filepaths of all the spectra in finder dirpath

    """
    filename_list = []
    for parent, dirnames, filenames in os.walk(dirpath):
        filename_list.extend([os.path.join(parent, filename)
                              for filename in filenames])
    n = len(filename_list)
    filename_list = filename_list[1:n+1]
    return filename_list
    

# for fucntions below, consider whether it is need to be here
# #############################################################################


def test_measure_line_index():
    filepath = walk_dir('')
    n = len(filepath)
    line_indx_star = [[]for i in range(3)]
    for i in range(n):
        line_indx = measure_line_index_loopfun(filepath[i])
        line_indx_star[0].append(line_indx[0])
        line_indx_star[1].append(line_indx[1])
        line_indx_star[2].append(line_indx[2])
    return line_indx_star


def get_equivalent_width(line_indx_star):
    EW = [[] for i in range(3)]
    n = len(line_indx_star[0])
    for i in range(3):
        for j in range(n):
            EW[i].append(line_indx_star[i][j]['EW_int'])
    return EW
    

def plot_equivalent_width_hist(EW_star):
    titles = ["5780", "5797", "6285"]
    fig, axes = plt.subplots(1, 3, figsize=(8, 8))
    for i in range(3):
        ax = axes[0, i]
        ax.hist(EW_star[i], facecolor='red', alpha=0.5)
        ax.set_xlabel('equivalent width')
        ax.set_ylabel('number')
        ax.set_title('Histogram of equivalent width_line'+titles[i])
        plt.tight_layout()
    plt.show()

    
def plot_line_indices(EW_star):
    titles = ["5780", "5797", "6285"]
    fig, axes = plt.subplots(3, 3, figsize=(64, 64))
    for i in range(3):
        for j in range(i+1):
            ax = axes[i, j]
            ax.set_title(titles[i]+" - "+titles[j], fontsize = 8)
            ax.set_ylabel(titles[i], fontsize=8)
            ax.set_xlabel(titles[j], fontsize=8)
            ax.plot(EW_star[j], EW_star[i], 'ob',markersize=3, alpha=0.5)
    plt.tight_layout()

# #############################################################################

# %% test
if __name__ == '__main__':
    # line_indx_star = test_measure_line_index()
    # EW_star = get_equivalent_width(line_indx_star)
    # plot_equivalent_width_hist(EW_star)
    # plot_line_indices(EW_star)
    test_()

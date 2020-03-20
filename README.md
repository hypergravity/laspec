## laspec
[![DOI](https://zenodo.org/badge/202476664.svg)](https://zenodo.org/badge/latestdoi/202476664) ![Upload Python Package](https://github.com/hypergravity/laspec/workflows/Upload%20Python%20Package/badge.svg)

Modules for basic operations on **LA**MOST **spec**tra, etc.

## author
Bo Zhang, [bozhang@nao.cas.cn](mailto:bozhang@nao.cas.cn)

## home page
- [https://github.com/hypergravity/laspec](https://github.com/hypergravity/laspec)
- [https://pypi.org/project/laspec/](https://pypi.org/project/laspec/)

## install
- for the latest **stable** version: `pip install -U laspec`
- for the latest **github** version: `pip install -U git+git://github.com/hypergravity/laspec`

## module structure
- **binning** \
    module for rebinning spectra
    - rebin(wave, flux, flux_err, mask): rebin spectra
- **ccf** \
    module for cross correlation function
    - xcorr_rvgrid: cross-correlation given an RV grid
    - xcorr: standard cross-correlation
    - sine_bell: a sine bell function
- **convolution** \
    module for spectral Gaussian convolution
    - conv_spec: capable to tackle arbitrary R_hi and R_lo but relatively slow
- **interpolation** \
    interpolation, but slow, please do not use.
    - Interp1q: use numpy.interp instead
- **lamost** \
    module for LAMOST spectra and files
    - lamost_filepath(planid, mjd, spid, fiberid)
    - lamost_filepath_med(planid, mjd, spid, fiberid)
    - sdss_filepath(plate, mjd, fiberid)
- **mrs** \
    MRS module
    - MrsSpec: MRS spectrum (B / R)
    - MrsEpoch: MRS epoch spectrum (B + R)
    - MrsFits(astropy.io.fits.HDUList): MRS fits reader
    - MrsSource(numpy.ndarray): MRS source constructor 
- **line_indices** \
    module to measure spectral line index (EW)
    - measure_line_index: measure line index (EW)
- **normalization** \
    module to normalize spectra
    - normalize_spectrum: a Python version of Chao's method
    - normalize_spectrum_iter: iterative normalization (recommended)
- **qconv** \
    quick convolution, designed for two cases:
    - conv_spec_Gaussian(wave, flux, R_hi=3e5, R_lo=2000): scalar resolution to scalar resolution instrumental broadening
    - conv_spec_Rotation(wave, flux, vsini=100., epsilon=0.6): stellar rotation broadening    
- **read_spectrum** \
    module to read LAMOST/SDSS spectra
    - read_spectrum(fp): read LAMOST low-res spectra
    - read_lamostms(fp): read LAMOST medium-res spcetra    
- **spec** \
    modules for operations on general spectra (deprecated)
    - Spec: spec class
- **wavelength** \
    module to convert wavelength between air and vacuum
    - wave_log10: log10 wavelength grid
    - vac2air: convert wavelength from vacuum to air
    - air2vac: convert wavelength from air to vacuum


## acknowledgements

...

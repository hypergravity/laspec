## laspec

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
    module for rebin spectra
- **ccf** \
    module for cross correlation function
- **convolution** \
    module for spectral Gaussian convolution (arbitrary case)
    - lamost.convolution.conv_spec
- **qconv** \
    quick convolution, designed for two cases:
    - lamost.qconv.conv_spec_Gaussian: scalar resolution to scalar resolution instrumental broadening
    - lamost.qconv.conv_spec_Rotation: stellar rotation broadening    
- **lamost** \
    module for LAMOST spectra and files
- **line_indices** \
    module to measure spectral line index (EW)
- **normalization** \
    module to normalize spectra
- **read_spectrum** \
    module to read LAMOST/SDSS spectra
- **spec** \
    modules for operations on general spectra
- **wavelength** \
    module to convert wavelength between air and vacuum


## acknowledgements

...

import numpy as np


class RVC:
    """
    radial velocity calibrator
    """
    star_weight = np.array([])

    def __init__(self, t, ra, dec, rv_obs, rv_obs_err,
                 rv_std=None, rv_std_err=None,
                 rv_val=None, rv_val_err=None):
        """

        Parameters
        ----------
        ra:
            R.A.
        dec:
            Dec.
        rv_obs:
            observed RV
        rv_obs_err:
            error of rv_obs
        rv_std:
            RV standard
        rv_std_err：
            error of rv_std
        rv_val:
            RV validation
        rv_val_err:
            error of rv_val
        """

        pass

# internal match:
# java -jar stilts.jar tmatch1 matcher=sky values="ra dec" params=1 action=identify in=test.csv ifmt=csv out=test.identify.csv ofmt=csv
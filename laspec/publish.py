import numpy as np


def format_float(t, float_max=1e20):
    """ simply format the float columns (set 1e20 to nan) """
    for colname in t.colnames:
        if isinstance(t[0][colname], float):
            # count the invalid elements
            ind_invalid = t[colname] >= float_max
            n_invalid = np.sum(ind_invalid)
            t[colname][ind_invalid] = np.nan
            print("setting {} elements to nan for column *{}*".format(n_invalid, colname))
    return



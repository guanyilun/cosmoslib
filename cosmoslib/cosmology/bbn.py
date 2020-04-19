"""BBN related"""

from cosmoslib.utils import context
from scipy import interpolate
import numpy as np


def Y_p(omega_b, Nnu):
    """Calculate BBN-standard nucleon number fraction by interpolating
    results from AlterBBN v1.4"""
    _omega_b, _delta_Nnu, _Yp = np.loadtxt(context.bbn_table, unpack=True, usecols=[0,2,4])
    intp = interpolate.bisplrep(_omega_b, _delta_Nnu, _Yp)
    return interpolate.bisplev(omega_b, Nnu-3.046, intp)

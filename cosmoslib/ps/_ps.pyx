import numpy as np
from .ps import Dl2Cl, Cl2Dl
cimport numpy as np
from libc.math cimport exp, log

cdef double Wb(double l, double fwhm):
    return exp(l*(l+1)*fwhm**2/(8*log(2)))

def covmat(np.ndarray[np.float64_t, ndim=2] ps, double noise, double fwhm, int l_min,
           int l_max, double f_sky, int prefactor=True):
    """Calculate the covariance matrix based on a model.

    Args:
        ps: power spectra
        noise: noise per pixel
        fwhm: beam size in degress (full width half minimum)
        l_min, l_max: range of ells
        f_sky: sky coverage fraction, 1 means full-sky coverage

    Returns:
        cov: a tensor of size [n_ell, n_ps, n_ps], for example with
             a lmax of 5000, the tensor size will be [5000, 4, 4]
    """
    cdef int i, n_ells
    cdef double l, wTinv, wPinv
    cdef double[::1] ells, ClTT, ClEE, ClBB, ClTE
    cdef np.ndarray[np.float64_t, ndim=3] cov
    cdef np.ndarray[np.uint8_t, cast=True] mask
    # assuming the beam is a gaussian beam with an ell dependent
    # beam size
    if prefactor:
        ps = Dl2Cl(ps, inplace=False)

    _ells = ps[:, 0]

     # calculate the noise parameter w^-1
    wTinv = noise**2
    wPinv = 2*wTinv

    mask = np.logical_and(_ells>=l_min, _ells<=l_max)

    # extract power spectra
    ells = ps[mask,0]
    ClTT = ps[mask,1]
    ClEE = ps[mask,2]
    ClBB = ps[mask,3]
    ClTE = ps[mask,4]

    # initialize empty covariance tensor. Since the covariance matrix
    # depends on ell, we will make a higher dimensional array [n_ell,
    # n_ps, n_ps] where the first index represents different ells, the
    # second and third parameters represents different power spectra
    n_ells = len(ells)
    cov = np.zeros([n_ells, 4, 4])

    for i in range(n_ells):
        l = ells[i]
        # T, T
        cov[i,0,0] = 2.0/(2*l+1)*(ClTT[i] + wTinv*Wb(l,fwhm))**2
        # E, E
        cov[i,1,1] = 2.0/(2*l+1)*(ClEE[i] + wPinv*Wb(l,fwhm))**2
        # B, B
        cov[i,2,2] = 2.0/(2*l+1)*(ClBB[i] + wPinv*Wb(l,fwhm))**2
        # TE, TE
        cov[i,3,3] = 1.0/(2*l+1)*(ClTE[i]**2 + (ClTT[i] + wTinv*Wb(l,fwhm))
                                  *(ClEE[i] + wPinv*Wb(l,fwhm)))
        # T, E
        cov[i,0,1] = cov[i,1,0] = 2.0/(2*l+1)*ClTE[i]**2
        # T, TE
        cov[i,0,3] = cov[i,3,0] = 2.0/(2*l+1)*ClTE[i]*(ClTT[i] +
                                                       wTinv*Wb(l,fwhm))
        # E, TE
        cov[i,1,3] = cov[i,3,1] = 2.0/(2*l+1)*ClTE[i]*(ClEE[i] +
                                                       wPinv*Wb(l,fwhm))
    # now we include the effect of partial sky coverage
    cov /= f_sky

    return ells, cov

import numpy as np
from .ps import Dl2Cl, Cl2Dl
from cython.parallel import prange

cimport cython
cimport numpy as np
from libc.math cimport exp, log

cdef extern from "complex.h":
    double creal(complex arg)
    double cimag(complex arg)

cdef double Wb(double l, double fwhm) nogil:
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

    for i in prange(n_ells, nogil=True):
        with cython.boundscheck(False):
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


def gen_ps_realization(np.ndarray[np.float64_t, ndim=2] ps, int prefactor=True):
    """Generate a random power spectra realization

    Args:
        ps: power spectra
        prefactor: true if ps is Dl

    Returns:
        ps realization: consistent with prefactor choice
    """
    cdef int i, l
    cdef double[:,::1] m_ps = np.zeros_like(ps)
    cdef np.ndarray[np.complex128_t, ndim=1] zeta1, zeta2, zeta3
    cdef double i_ClTT, i_ClEE, i_ClBB, i_ClTE
    cdef double[::] ells, ClTT, ClEE, ClBB, ClTE
    cdef np.ndarray[np.complex128_t, ndim=1] aTlm, aElm, aBlm
    # first make a copy to make sure we don't affect the original
    if prefactor:
        ps = Dl2Cl(ps, inplace=False)
    ells, ClTT, ClEE, ClBB, ClTE = ps[:,0], ps[:,1], ps[:,2], ps[:,3], ps[:,4]

    # define empty arrays to hold the generated power spectra
    m_ps[:,0] = ells

    # this is certainly slow, but i don't need to run this very often
    # so it's fine to leave it like this. in principle i don't need to
    # keep all the negative part as well. These can be improved if
    # performance becomes an issue
    for i in range(len(ells)):
        l = int(ells[i])

        # generate gaussian random complex numbers with unit variance
        zeta1 = np.random.randn(l+1)*np.exp(1j*np.random.rand(l+1)*2*np.pi)
        zeta2 = np.random.randn(l+1)*np.exp(1j*np.random.rand(l+1)*2*np.pi)
        zeta3 = np.random.randn(l+1)*np.exp(1j*np.random.rand(l+1)*2*np.pi)

        # for m=0, zeta has to be real
        zeta1[0] = abs(zeta1[0])
        zeta2[0] = abs(zeta2[0])
        zeta3[0] = abs(zeta3[0])

        # generate alm
        aTlm = zeta1 * ClTT[i]**0.5
        aElm = zeta1 * ClTE[i] / (ClTT[i])**0.5 + zeta2*(ClEE[i] - ClTE[i]**2/ClTT[i])**0.5
        aBlm = zeta3 * ClBB[i]**0.5

        i_ClTT = (aTlm[0]**2 + 2*(np.sum(np.abs(aTlm[1:])**2)))/(2*l+1)
        i_ClEE = (aElm[0]**2 + 2*(np.sum(np.abs(aElm[1:])**2)))/(2*l+1)
        i_ClBB = (aBlm[0]**2 + 2*(np.sum(np.abs(aBlm[1:])**2)))/(2*l+1)
        i_ClTE = (aTlm[0]*aElm[0] + 2*(np.sum(np.conj(aTlm[1:])*aElm[1:])))/(2*l+1)

        # assign the new values to the new array
        m_ps[i,1] = i_ClTT
        m_ps[i,2] = i_ClEE
        m_ps[i,3] = i_ClBB
        m_ps[i,4] = i_ClTE

    if prefactor:
        return Cl2Dl(m_ps, inplace=True)
    else:
        return m_ps

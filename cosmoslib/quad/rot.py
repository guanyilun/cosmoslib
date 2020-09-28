"""Reconstruction of rotational field"""

import numpy as np
import healpy as hp

from cosmoslib.utils.glquad import gauss_legendre_quadrature


#----------------------------------------------------------------------
# reconstruction of rotational field
#
#----------------------------------------------------------------------


def qeb(lmax, rlmin, rlmax, fCE, Elm, Blm, alm, nside=None):
    """Reconstructing pol rotation angle from the EB quadratic estimator.
    adapted from the fortran version in cmblensplus

    Parameters:
    -----------
      lmax (int)         : Maximum multipole of output lensing potential alms
      rlmin/rlmax (int)  : Minimum/Maximum multipole of CMB for reconstruction
      fCE [l] (double)   : EE spectrum, with bounds (0:rlmax)
      Elm [l,m] (dcmplx) : Inverse-variance filtered E-mode alm, with bounds (0:rlmax,0:rlmax)
      Blm [l,m] (dcmplx) : Inverse-variance filtered B-mode alm, with bounds (0:rlmax,0:rlmax)
      nside (int)        : Nside for the convolution calculation, default to lmax

    Returns:
    --------
      alm [l,m] (dcmplx) : Rotation angle alm, with bounds (0:lmax,0:lmax)

    """
    if not nside: nside=lmax
    print(f"Calculating qEB rotation estimator with nside={nside}")
    npix = 12*nside**2
    # move alm into a right shape for alm2map_spin
    # Blm -> Bmap
    alm = np.zeros((2,)+Blm.shape)
    alm[1,...] = Blm
    Bmap = hp.alm2map_spin(alm, nside, 2, rlmax, rlmax)  # A
    # Elm -> Emap
    alm = np.zeros((2,)+Blm.shape)
    alm[0,...] = hp.almxfl(Elm,fCE)  # wiener-filtering
    Emap = hp.alm2map_spin(alm, nside, 2, rlmax, rlmax)  #A2
    # Map-space operation
    omap = Bmap[0]*Emap[1] - Bmap[1]*Emap[0]
    # convert back to alm space
    oalm = -2*hp.map2alm(omap, lmax, lmax)
    return oalm


#----------------------------------------------------------------------
# rotational field reconstruction noise
#
#----------------------------------------------------------------------
def qeb_nlaa_simple(theory, noise):
    """Convenient wrapper of the underlying function"""
    obs = (theory + noise).remove_prefactor()
    lmax = obs.lmax
    theory = theory.resample(obs.ell).remove_prefactor()
    return qeb_nlaa(lmax, theory.EE, theory.BB, obs.EE, obs.BB)


def qeb_nlaa(lmax_a, clee, clbb, oclee, oclbb):
    """reconstruction noise from EB"""

    lmax_e = len(clee) - 1
    lmax_b = len(clbb) - 1

    assert(lmax_b >= 2)
    assert(lmax_e >= 2)
    assert(lmax_a >= 1)
    assert(lmax_b <= (lmax_e+lmax_a-1))

    gl = gauss_legendre_quadrature(int((lmax_e + lmax_a + lmax_b)*0.5) + 1)

    ls = np.arange(0, lmax_e+1, dtype=np.double)
    WA = (2*ls+1)/(4*np.pi) * (1./oclee)
    ls = np.arange(0, lmax_b+1, dtype=np.double)
    WB = (2*ls+1)/(4*np.pi) * (clbb**2/oclbb)

    zeta_A_22_p = gl.cf_from_cl(2,  2, WA)
    zeta_A_22_m = gl.cf_from_cl(2, -2, WA)
    zeta_B_22_p = gl.cf_from_cl(2,  2, WB)
    zeta_B_22_m = gl.cf_from_cl(2, -2, WB)

    nlaa = gl.cl_from_cf(lmax_a, 0, 0,
                         zeta_A_22_p*zeta_B_22_p +
                         zeta_A_22_m*zeta_B_22_m)

    return 1./(4*np.pi*nlaa)

"""Reconstruction of rotational field"""

import numpy as np
from cosmoslib.utils.glquad import gauss_legendre_quadrature

#----------------------------------------------------------------------
# rotational field reconstruction noise
#
#----------------------------------------------------------------------

def qeb_nlaa(lmax_a, clee, clbb, oclee, oclbb):
    """reconstruction noise from EB"""

    lmax_e = len(clee) - 1
    lmax_b = len(clbb) - 1

    assert( lmax_b >= 2 )
    assert( lmax_e >= 2 )
    assert( lmax_a >= 1 )
    assert( lmax_b <= (lmax_e+lmax_a-1) )

    gl = gauss_legendre_quadrature( int((lmax_e + lmax_a + lmax_b)*0.5)+1 )

    ls = np.arange(0, lmax_e+1, dtype=np.double)
    WA = (2*ls+1)/(4*np.pi) * (1./oclee)
    ls = np.arange(0, lmax_b+1, dtype=np.double)
    WB = (2*ls+1)/(4*np.pi) * (clbb**2/oclbb)

    zeta_A_22_p = gl.cf_from_cl( 2,  2, WA )
    zeta_A_22_m = gl.cf_from_cl( 2, -2, WA )
    zeta_B_22_p = gl.cf_from_cl( 2,  2, WB )
    zeta_B_22_m = gl.cf_from_cl( 2, -2, WB )

    nlaa = gl.cl_from_cf( lmax_a, 0, 0,
                          zeta_A_22_p*zeta_B_22_p +
                          zeta_A_22_m*zeta_B_22_m)

    return 1./(4*np.pi*nlaa)

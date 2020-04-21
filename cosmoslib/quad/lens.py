"""calculate lensing normalization / reconstruction noises from various estimators"""

import numpy as np
from cosmoslib.utils.glquad import gauss_legendre_quadrature

#----------------------------------------------------------------------
# lensing reconstruction noise
#
#----------------------------------------------------------------------

def qeb_nlpp(lmax_p, clbb, clee, nleb):
    """Lensing reconstruction using EB.
    Taken from https://github.com/marius311/cmblensing.
    Equations come from arXiv:1010.0048v2.
    TODO: need to verify the formula used here: is it nleb?
    """

    lmax_e = len(clee) - 1
    lmax_b = len(clbb) - 1

    assert( lmax_b >= 2 )
    assert( lmax_e >= 2 )
    assert( lmax_p >= 1 )
    assert( lmax_b <= (lmax_e+lmax_p-1) )
    gl = gauss_legendre_quadrature( int((lmax_e + lmax_p + lmax_b)*0.5)+1 )

    ls = np.arange(0, lmax_e+1, dtype=np.double)
    cl_en = clee**2/(clee+nleb)
    cl_en33 = (2*ls+1)/(4*np.pi)*cl_en*(ls-2)*(ls+3)
    cl_en31 = (2*ls+1)/(4*np.pi)*cl_en*np.sqrt( (ls-1)*(ls+2)*(ls-2)*(ls+3) )
    cl_en11 = (2*ls+1)/(4*np.pi)*cl_en*(ls-1)*(ls+2)

    ls = np.arange(0, lmax_b+1, dtype=np.double)
    cl_bn22 = (2*ls+1)/(4*np.pi)*(1./(clbb+nleb))

    zeta_en33_p = gl.cf_from_cl( 3,  3, cl_en33 )
    zeta_en33_m = gl.cf_from_cl( 3, -3, cl_en33 )
    zeta_en31_p = gl.cf_from_cl( 3,  1, cl_en31 )
    zeta_en31_m = gl.cf_from_cl( 3, -1, cl_en31 )
    zeta_en11_p = gl.cf_from_cl( 1,  1, cl_en11 )
    zeta_en11_m = gl.cf_from_cl( 1, -1, cl_en11 )

    zeta_bn22_p = gl.cf_from_cl( 2,  2, cl_bn22 )
    zeta_bn22_m = gl.cf_from_cl( 2, -2, cl_bn22 )

    nlpp_out_p = gl.cl_from_cf( lmax_p, 1, 1,
                                zeta_en33_p*zeta_bn22_p -
                                2.*zeta_en31_m*zeta_bn22_m +
                                zeta_en11_p*zeta_bn22_p)

    nlpp_out_m =  gl.cl_from_cf( lmax_p, 1, -1,
                                 zeta_en33_m*zeta_bn22_m -
                                 2.*zeta_en31_p*zeta_bn22_p +
                                 zeta_en11_m*zeta_bn22_m)

    return 1./(np.pi/4.*(ls* (ls+1) ) * ( nlpp_out_p - nlpp_out_m ))


def qtt_nlpp(lmax_p, cltt, cltt, nltt):
    """Reconstruction noise from TT estimator, refer Eq. 45-48
    in arXiv:1010.0048v2.

    """
    lmax_t = len(cltt) - 1

    # initialize gl quadrature
    gl = gauss_legendre_quadrature(int(lmax_t*2+lmax_p))

    ls = np.arange(0, lmax_t+1, dtype=np.double)

    # common factors
    llp1 = ls*(ls+1)
    div_dl = 1/(cltt+nltt)
    cl_div_dl = cltt/(cltt+nltt)

    # get zeta terms
    zeta_00   = gl.cf_from_cl(0,  0, (2*ls+1)/(4*np.pi) * div_dl)
    zeta_01_p = gl.cf_from_cl(0,  1, (2*ls+1)/(4*np.pi) * np.sqrt(llp1)*cl_div_dl)
    zeta_01_m = gl.cf_from_cl(0, -1, (2*ls+1)/(4*np.pi) * np.sqrt(llp1)*cl_div_dl)
    zeta_11_p = gl.cf_from_cl(1,  1, (2*ls+1)/(4*np.pi) * llp1*cltt*cl_div_dl)
    zeta_11_m = gl.cf_from_cl(1, -1, (2*ls+1)/(4*np.pi) * llp1*cltt*cl_div_dl)

    nlpp_term_1 = gl.cl_from_cf(lmax_p, -1, -1,
                                zeta_00*zeta_11_p - zeta_01_p**2)
    nlpp_term_2 = gl.cl_from_cf(lmax_p,  1, -1,
                                zeta_00*zeta_11_m - zeta_01_p*zeta_01_m)

    return 1./(np.pi*llp1*(nlpp_term_1 + nlpp_term_2))

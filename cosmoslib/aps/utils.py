import healpy as hp


def alm2cl_healpy(alm1, alm2=None, lmax=None, mmax=None, lmax_out=None, nspec=None):
    """simple wrapper for healpy.sphfunc.alm2cl that
    returns a PS object"""
    cl = hp.sphtfunc.alm2cl(alm1, alm2, lmax, mmax, lmax_out, nspec)
    ps = PS(cl, order=('TT','EE','BB','TE','EB','TB'), prefactor=False)
    return ps

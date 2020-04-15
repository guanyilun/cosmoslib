from libc.math cimport log
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double exact_like(double[:] ells, double[:] ClTT, double[:] ClEE, double[:] ClBB,
                        double[:] ClTE, double[:] DlTT, double[:] DlEE, double[:] DlBB,
                        double[:] DlTE, double lmin, double lmax, double f_sky):
    """internal function to calculate exact-likelihood
    Args:
        ells, Cl{TT,EE,BB,TE}: theory c_\ell
        Dl{TT,EE,BB,TE}: data c_\ell
        l{min,max}: range of ell to consider
        f_sky: sky coverage
    Returns:
        log-likelihood
    """
    cdef double chi2, like, det, dof, l
    cdef int i

    chi2 = 0
    for i in range(len(ells)):
        l = ells[i]
        if l < lmin: continue
        if l > lmax: continue
        # T, E
        det = ClTT[i]*ClEE[i]-ClTE[i]**2
        dof = f_sky*(2*l+1)
        chi2 += dof*(1./det*(DlTT[i]*ClEE[i]-2*DlTE[i]*ClTE[i]+DlEE[i]*ClTT[i]) + \
                     log(det/(DlTT[i]*DlEE[i]-DlTE[i]**2))-2)
        # B
        chi2 += dof*(1./ClBB[i]*DlBB[i]+log(ClBB[i]/DlBB[i])-1)
    like = -0.5*chi2
    return like

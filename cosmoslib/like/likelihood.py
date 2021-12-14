"""various likelihood calculation"""
import numpy as np
# from cosmoslib.ps import Dl2Cl, resample
# from cosmoslib.like import _likelihood as cython

class ExactLikelihood:
    def __init__(self, ps_data, noise, f_sky=1., lmin=None, lmax=None):
        """Calculate Exact likelihood (wishart likelihood).

        Args:
            ps_data (PS): power spectrum object from data
            noise (Noise): noise object
            f_sky: sky coverage
        """
        self.ps_data = ps_data
        self.noise = noise
        self.f_sky = f_sky
        self.lmin = lmin
        self.lmax = lmax

    def __call__(self, ps_theory):
        """Assuming ps_theory is a power spectrum (PS) object"""
        ps_resample = ps_theory.resample(self.ps_data.ell)
        ps_w_noise = (ps_theory + self.noise).resample(self.ps_data.ell)
        ps_data = self.ps_data.resample(ps_w_noise.ell)
        cl = ps_w_noise.remove_prefactor()
        dl = ps_data.remove_prefactor()
        lmin = self.lmin if self.lmin else ps_data.lmin
        lmax = self.lmax if self.lmax else ps_data.lmax
        return cython.exact_like(cl['ell'],cl['TT'],cl['EE'],cl['BB'],cl['TE'],
                                 dl['TT'],dl['EE'],dl['BB'],dl['TE'],lmin,
                                 lmax,self.f_sky)


class GaussianLikelihood:
    def __init__(self, ps_data, noise, cov, f_sky=1., lmin=None, lmax=None):
        """Calculate Gaussian likelihood with fixed covariance

        Args:
            ps_data (PS): power spectrum object from data
            noise (Noise): noise object
            f_sky: sky coverage
            cov: inverse covmat of Covmat class
        """
        self.ps_data = ps_data
        self.noise = noise
        self.cov = cov
        self.f_sky = f_sky
        self.icov = cov.inv()
        self.lmin = lmin
        self.lmax = lmax

    def __call__(self, ps_theory):
        # for varied covariance matrix
        # ell, cov = ps_theory.covmat(self.noise, self.f_sky)
        ell, icov = self.icov.ell, self.icov.cov
        ps_w_noise = (ps_theory+self.noise).resample(ell)
        ps_data = self.ps_data.resample(ell)
        cl = ps_w_noise.remove_prefactor().values
        dl = ps_data.remove_prefactor().values
        chi2 = 0
        for i in range(len(ell)):
            l = ell[i]
            # respect the boundary if that's set
            if self.lmin:
                if l < self.lmin: continue
            if self.lmax:
                if l > self.lmax: continue
            # note that there shouldn't not be an absolute value here
            err = cl[i,1:]-dl[i,1:]
            # using a constant cov, so log_det is not needed
            # log_det = -np.log(np.linalg.det(icov[i,:,:]))
            chi2 += np.einsum("ij,i,j", icov[i,:,:], err, err)*self.f_sky # + log_det
        like = -0.5*chi2
        return like



# legacy function: leave it here for backwards compatibility
def exact_likelihood(ps_theory, ps_data, nl, f_sky=1., prefactor=True):
    """Calculate the exact likelihood based on the T, E, B, TE power
    spectra.

    Parameters:
    ------------
    ps_theory: theory power spectra
    ps_data: data power spectra
    nl: noise spectra
    f_sky: fraction of sky covered. This is added as an effective
           reduction in the dof in chi-square calculation, default
           to 1.
    prefactor: boolean. True if Dls are provided, otherwise False.

    Return:
    --------
    log-likelihood: float

    """

    ell = ps_data[:, 0]

    # resample the theory curves to match the observation
    ps_resample = resample(ps_theory, ell)

    if prefactor:
        ps_resample = Dl2Cl(ps_resample)
        ps_data = Dl2Cl(ps_data)
        nl = Dl2Cl(nl)

    cls = ps_resample[:, 1:] + nl[:, 1:]
    dls = ps_data[:, 1:]

    chi2 = 0
    for i, l in enumerate(ell):
        cl = cls[i, :]   # cl theory
        dl = dls[i, :]  # cl data

        # T, E
        det = cl[0]*cl[1]-cl[3]**2
        dof = f_sky*(2*l+1)

        chi2 += dof*(1./det*(dl[0]*cl[1]-2*dl[3]*cl[3]+dl[1]*cl[0])+\
                     np.log(det/(dl[0]*dl[1]-dl[3]**2))-2)
        # B
        chi2 += dof*(1./cl[2]*dl[2]+np.log(cl[2]/dl[2])-1)

    like = -0.5*chi2

    return like

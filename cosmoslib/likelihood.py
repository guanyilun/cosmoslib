"""various likelihood calculation"""
import numpy as np
from .ps import Dl2Cl, resample


class ExactLikelihood:
    def __init__(self, ps_data, noise, f_sky=1.):
        """Calculate Exact likelihood (wishart likelihood).

        Args:
            ps_data (PS): power spectrum object from data
            noise (Noise): noise object
            f_sky: sky coverage
        """
        self.ps_data = ps_data
        self.noise = noise
        self.f_sky = f_sky

    def __call__(self, ps_theory):
        """Assuming ps_theory is a power spectrum (PS) object"""
        ps_resample = ps_theory.resample(self.ps_data.ell)
        ps_w_noise = ps_resample + noise
        cl = ps_w_noise.ps
        dl = self.ps_data.ps

        chi2 = 0
        for i, l in enumerate(psd.ell):
            # T, E
            det = cl['TT'][i]*cl['EE'][i]-cl['TE'][i]**2
            dof = self.f_sky*(2*l+1)

            chi2 += dof*(1./det*(dl['TT'][i]*cl['EE'][i]-2*dl['TE'][i]*cl['TE']['i'] + \
                                 dl['EE'][i]*cl['TT'][i]) + \
                         np.log(det/(dl['TT'][i]*dl['EE'][i]-dl['TE'][i]**2))-2)
            # B
            chi2 += dof*(1./cl['BB'][i]*dl['BB'][i]+np.log(cl['BB'][i]/dl['BB'][i])-1)

        like = -0.5*chi2


    @staticmethod
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

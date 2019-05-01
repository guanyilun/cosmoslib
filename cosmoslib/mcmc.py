"""This script will contain useful functions and class for the
mcmc study."""

import numpy as np
import emcee

from cosmoslib.ps import Dl2Cl

class MCMC(object):

    base_params = {}
    fit_params = {}
    fit_keys = []  # specify an ordering of the param keys
    cosmology = None
    sampler = None

    def __init__(self, ps_data=None, n_walkers=2, f_sky=1., initial_delta):
        self.f_sky = f_sky
        self.ps_data = ps_data
        self.n_walkers = n_walkers
        self.initial_delta = initial_delta

    def set_params(self, params):
        """This method assigns parameters to the MCMC algorithm"""
        for k in params:
            if type(params[k]) is list:
                self.fit_params[k] = params[k]
            elif type(params[k]) is tuple:
                self.fit_params[k] = list(params[k])
            else:
                # otherwise it's a number -> base param
                self.base_params[k] = params[k]

        # update fit_keys
        self.fit_keys = [*fit_params]

        # update cosmology if it exists
        if self.cosmology:
            self.cosmology.set_model_params(base_params)

    def lnprior(self, theta):
        """This method looks at the fit params and transform it
        into priors"""
        for i, k in enumerate(self.fit_keys):
            lower = self.fit_params[k][0]
            fid = self.fit_params[k][1]
            upper = self.fit_params[k][2]

            p = theta[i]
            if p < lower:
                return -np.inf
            elif p > upper:
                return -np.inf
        return 0

    @staticmethod
    def exact_likihood(ps_theory, ps_data, nl, f_sky=1., prefactor=True):
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

    def set_cosmology(self, cm):
        """Supply a cosmology object that contains the base parameters"""
        self.cosmology = cm

        # update the base parameters
        self.cosmology.set_model_params(self.base_params)

    def generate_theory(self, theta):
        """This function takes the parameters and generate a theory power spectra"""
        # initialize an empty dict to store the model parameters
        model_params = {}
        for i, k in enumerate(self.fit_keys):
            model_params[k] = theta[i]

        self.cosmology.set_model_params(model_params)

        return self.cosmology.run()

    def lnprob(self, theta):
        ps_theory = self.generate_theory(theta)

        prior = self.lnprior(theta)
        if np.isfinite(prior):
            like = MCMC.exact_likihood(ps_theory, self.ps_data, self.f_sky)
            return prior + like
        else:
            return -np.inf

    def set_f_sky(self, f_sky):
        """Set the effective f_sky. It will be used as
        an effective reduction in the number of degrees of
        freedom"""
        self.f_sky = f_sky

    def set_ps_data(self, ps_data):
        """Set the data power spectra"""
        self.ps_data = ps_data

    def run(self, N, pos0=None):
        """Run the mcmc sampler with an ensemble sampler from emcee

        Parameters:
        ------------
        N: number of samples to be made for each
        pos0: initial positions, default to use built-in initial pos0
              generator, but it can be supplied manually

        """
        ndim = len(self.fit_keys)
        if not pos0:
            pos0 = self.generate_initial_pos()

        sampler = emcee.EnsembleSampler(self.n_walkers, ndim, self.lnprob)
        sampler.run_mcmc(pos0, N)

        self.sampler = sampler
        return self.sampler

    def generate_initial_pos(self):
        """Generate the initial position for the mcmc"""
        pos0 = np.array([self.fit_params[key][1] for key in self.fit_keys])
        ndim = len(self.fit_keys)
        pos0 += np.random.randn(ndim)*self.initial_delta*pos0
        return pos0
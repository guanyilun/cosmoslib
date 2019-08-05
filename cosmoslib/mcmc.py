"""This script will contain useful functions and class for the
mcmc study."""

import numpy as np
import emcee
import dill as pickle

from cosmoslib.ps import Dl2Cl, resample

class MCMC(object):

    def __init__(self, ps_data=None, N_l=None, n_walkers=2, f_sky=1.,
                 initial_delta=0.01, save_samples=100, checkpoint_file=None):
        self.f_sky = f_sky
        self.ps_data = ps_data
        self.N_l = N_l
        self.n_walkers = n_walkers
        self.initial_delta = initial_delta
        self.sampler = None
        self.cosmology = None
        self.base_params = {}
        self.fit_params = {}
        self.fit_keys = []
        self.save_samples = save_samples
        self._counter = 0
        self.checkpoint_file = checkpoint_file

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
        self.fit_keys = [*self.fit_params]

        # update cosmology if it exists
        if self.cosmology:
            self.cosmology.set_model_params(self.base_params)

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
        return 0.

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
        try:
            results = self.cosmology.full_run()
        except Exception as e:
            print("%s occurred, loglike=-np.inf" % type(e))
            import traceback; traceback.print_exc()
            return None, True  # the second return refers to error

        return results, False

    def checkpoint(self):
        # np.save(self.checkpoint_file, self.sampler.chain)
        # test saving the object
        print("Checkpointing...")
        # pickle.dump(self, open(self.checkpoint_file, "wb"))
        pickle.dump(self, open(self.checkpoint_file, "wb"))

    def lnprob(self, theta):
        # update counter
        self._counter += 1
        if self._counter % self.save_samples == 0:
            self.checkpoint()

        ps_theory, err = self.generate_theory(theta)
        # if there is an error, reject this data point
        if err:
            return -np.inf

        # now i trust that there is no error
        prior = self.lnprior(theta)
        if np.isfinite(prior):
            like = MCMC.exact_likelihood(ps_theory, self.ps_data,
                                         self.N_l, self.f_sky)
            print("Parameter: %s\t loglike: %.2f" % (theta, like))
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

    def run(self, N, pos0=None, resume=False):
        """Run the mcmc sampler with an ensemble sampler from emcee

        Parameters:
        ------------
        N: number of samples to be made for each
        pos0: initial positions, default to use built-in initial pos0
              generator, but it can be supplied manually
        resume: whether to resume from the checkpoint file

        """
        self._counter = 0
        ndim = len(self.fit_keys)

        if self.n_walkers < 2*ndim:
            self.n_walkers = 2*ndim
            print("Warning: n_walkers too small, use %d instead..." \
                  % self.n_walkers)

        if not pos0:
            if not resume:
                pos0 = self.generate_initial_pos()
            else:
                print("Resuming from checkpoint file...")
                pos0 = self.get_initial_pos_from_ckp()

        # check if n_walker satisfy the requirement that it
        # has to be even and more than 2*ndim

        sampler = emcee.EnsembleSampler(self.n_walkers, ndim,
                                        self.lnprob)
        self.sampler = sampler
        sampler.run_mcmc(pos0, N)

        return self.sampler

    def generate_initial_pos(self):
        """Generate the initial position for the mcmc"""
        pos0 = []
        ndim = len(self.fit_keys)
        for i in range(self.n_walkers):
            pos = np.array([self.fit_params[key][1] for key in
                            self.fit_keys])

            pos += np.random.randn(ndim)*self.initial_delta*pos
            pos0.append(pos)

        return pos0

    def get_initial_pos_from_ckp(self):
        """Get the initial position from an unfinished chain"""
        with open(self.checkpoint_file, "rb") as f:
            data = pickle.load(f)
        n_existing = data.sampler.iterations-1
        initial_pos = data.sampler.chain[:,n_existing-1,:]
        del data
        return initial_pos

    def reset_params(self):
        self.base_params = {}
        self.fit_params = {}
        self.fit_keys = []  # specify an ordering of the param keys

    def get_bf_params(self):
        # get best fit parameters
        whmax = self.sampler.flatlnprobability.argmax()
        bf_values = self.sampler.flatchain[whmax,:]

        bf_params = {}
        for i, k in enumerate(self.fit_keys):
            bf_params[k] = bf_values[i]

        return bf_params

    def get_bf_cosmology(self):
        import copy

        bf_params = self.get_bf_params()
        cosmology = copy.deepcopy(self.cosmology)

        cosmology.set_model_params(bf_params)
        return cosmology


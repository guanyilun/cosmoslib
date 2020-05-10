"""This script contains wrapper class that interacts with all
components of the library to perform analysis such as fisher matrix
estimation, etc. It makes use of cambex: which is a good wrapper for
run camb the old fashioned way when there is no good way to interact
with python otherwise, such as for legecy codes that modify camb.

"""

from .cambex import CambSession
from scipy.interpolate import interp1d
import numpy as np
from cosmoslib.utils import load_context
from cosmoslib.ps import PS


class Cosmology(object):
    """Basic cosmology class based on camb"""
    def __init__(self, camb_bin=None, base_params=None,
                 model_params={}, rank=None, legacy=False, output_dir=None):
        # attempt to load supplied parameters from context
        context = load_context()
        if context:
            if not camb_bin:
                camb_bin = context['camb']['camb_bin']
                legacy = context['camb'].getboolean('camb_legacy', True)
            if not output_dir:
                output_dir = context['camb']['output_dir']
        self.camb = CambSession(camb_bin=camb_bin, rank=rank,
                                output_dir=output_dir)
        self.base_params = base_params
        self.model_params = model_params
        self.mode = "TFF"  # scaler only
        self.target_spectrum = lambda x: x.totCls
        self.legacy = legacy  # old version

    def set_base_params(self, params):
        """Set the baseline parameters"""
        self.base_params = params

    def set_model_params(self, params):
        """Set model specific parameters
        Args:
            params: a dictionary with the fiducial model that one is
            interested in.
        Example:
            cm.set_model_params({"initial_ratio": 0.0042})
        """
        self.model_params.update(params)

    def _populate_params(self, exclude_list=[]):
        """A translator function that make it convenient to define alias
        or other consistency relation between the parameters"""
        for k, v in self.model_params.items():
            # have an option to skip certain keys
            if k in exclude_list:
                continue
            # define look up rules here
            if k == 'r_t':
                if self.legacy:
                    self.camb.ini.set('initial_ratio(1)', v)
                else:
                    self.camb.ini.set('initial_ratio', v)
            elif k == 'initial_ratio':
                if self.legacy:
                    self.camb.ini.set('initial_ratio(1)', v)
                else:
                    self.camb.ini.set(k, v)
            elif k == 'h0':
                self.camb.ini.set('hubble', v * 100.)
            elif k == 'log1e10As':
                if self.legacy:
                    self.camb.ini.set('scalar_amp(1)', np.exp(v)*1E-10)
                else:
                    self.camb.ini.set('scalar_amp', np.exp(v)*1E-10)
            elif k == 'scalar_amp':
                if self.legacy:
                    self.camb.ini.set('scalar_amp(1)', v)
                else:
                    self.camb.ini.set(k, v)
            elif k == 'n_s':
                if self.legacy:
                    self.camb.ini.set('scalar_spectral_index(1)', v)
                else:
                    self.camb.ini.set('scalar_spectral_index',  v)
            elif k == 'scalar_spectral_index':
                if self.legacy:
                    self.camb.ini.set('scalar_spectral_index(1)', v)
                else:
                    self.camb.ini.set(k, v)
            elif k == 'tau':
                self.camb.ini.set('re_optical_depth', v)
            elif k == 'B0':
                self.camb.ini.set('magnetic_amp', v)  # nG
            elif k == 'n_B':
                self.camb.ini.set('magnetic_ind', v)
            elif k == 'lrat':
                self.camb.ini.set('magnetic_lrat', v)
            # otherwise it's a direct translation
            else:
                self.camb.ini.set(k, v)

    def set_mode(self, mode):
        """Set the mode of interests here, the mode should be a
        three character string with T or F corresponding to
        scaler, vector and tensor mode, default to TFF"""
        self.mode = mode

        self.set_model_params({
            "get_scalar_cls": self.mode[0],
            "get_vector_cls": self.mode[1],
            "get_tensor_cls": self.mode[2]
        })

    def run(self):
        """Run the cosmology model to get power spectra"""
        # define base parameters
        self.camb.set_base_params(self.base_params)

        # pass all model params to camb session
        self._populate_params()

        # run camb sessions
        self.camb.run()

        # load power spectra into memory
        self._load_ps()

        # clean-up the folder to avoid reading wrong files
        self.camb.cleanup()

        # return a user defined target spectrum
        target_spec = self.target_spectrum(self)
        # wrap as a PS object
        return PS(target_spec, prefactor=True)

    def full_run(self):
        """This will be the method called by MCMC sampler. It can be the same
        or different to the run method. By default it's the same as self.run()
        """
        return self.run()

    def _load_ps(self):
        try:
            self.scalarCls = self.camb.load_scalarCls().values.T
        except OSError:
            self.scalarCls = None

        try:
            self.vectorCls = self.camb.load_vectorCls().values.T
        except OSError:
            self.vectorCls = None

        try:
            self.tensorCls = self.camb.load_tensorCls().values.T
        except OSError:
            self.tensorCls = None

        try:
            self.lensedCls = self.camb.load_lensedCls().values.T
        except OSError:
            self.lensedCls = None

        try:
            self.totCls = self.camb.load_totCls().values.T
        except OSError:
            self.totCls = None

        try:
            self.lensedtotCls = self.camb.load_lensedtotCls().values.T
        except OSError:
            self.lensedtotCls = None

    def fisher_matrix(self, covmat, targets=[], ratio=0.01, verbose=False):
        """Calculate the fisher matrix for a given covariance matrix

        Args:
            cov: covariance matrix lx4x4
            targets: targeting cosmological parameters as a list
            ratio: percentage step size to estimate derivatives
        """
        if len(targets)==0:
            raise ValueError("No targets found, what do you want?")

        ells, cov = covmat.ell, covmat.cov
        model = self.model_params

        dCldp_list = []
        for p in targets:
            if not p in model.keys():
                print("Warning: %s not found in model, skipping..." % p)
                continue
            if verbose: print(f"Varying {p}...")
            # make copies of model for variation of parameters
            new_model_m1 = model.copy()
            new_model_m2 = model.copy()
            new_model_p1 = model.copy()
            new_model_p2 = model.copy()

            # step size
            h = model[p]*ratio

            new_model_m2[p] = model[p] - h
            new_model_m1[p] = model[p] - 0.5*h
            new_model_p1[p] = model[p] + 0.5*h
            new_model_p2[p] = model[p] + h

            if verbose: print("-> Calc m2...")
            self.set_model_params(new_model_m2)
            ps_m2 = self.full_run()

            if verbose: print("-> Calc m1...")
            self.set_model_params(new_model_m1)
            ps_m1 = self.full_run()

            if verbose: print("-> Calc p1...")
            self.set_model_params(new_model_p1)
            ps_p1 = self.full_run()

            if verbose: print("-> Calc p2...")
            self.set_model_params(new_model_p2)
            ps_p2 = self.full_run()

            # calculate differenciations to p
            dCldp = (4.0/3.0*(ps_p1-ps_m1) - 1.0/6.0*(ps_p2-ps_m2))/h
            # interpolate it into the given ells
            dCldp = dCldp.resample(ells).remove_prefactor()
            # store it to a list
            dCldp_list.append(dCldp)

        n_params = len(dCldp_list)
        alpha = np.zeros([n_params, n_params])
        n_ell = cov.shape[0]
        if verbose: print("Calc alpha...")
        for i in range(n_params):
            for j in range(n_params):
                for k in range(n_ell):
                    cl_i = dCldp_list[i].values[k,1:]
                    cl_j = dCldp_list[j].values[k,1:]
                    alpha[i,j] += np.einsum('i,ij,j', cl_i, np.linalg.inv(cov[k,:,:]), cl_j)

        return alpha

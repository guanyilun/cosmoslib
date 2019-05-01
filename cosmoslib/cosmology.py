"""This script contains wrapper class that interacts with all
components of the library to perform analysis such as fisher matrix
estimation, etc.
"""

from .cambex import CambSession
from scipy.interpolate import interp1d
import numpy as np


class Cosmology(object):

    target_spectrum = lambda x: x.totCls
    
    def __init__(self, camb_bin=None, base_params=None, model_params={}):
        self.camb = CambSession(camb_bin=camb_bin)
        self.base_params = base_params
        self.model_params = model_params
        self.mode = "TFF"  # scaler only

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
        self._update_model_params(params)

    def _update_model_params(self, params):
        """A translator function that make it convenient to define alias
        or other consistency relation between the parameters"""
        for k, v in params.items():
            # define look up rules here
            if k == 'r':
                self.model_params['initial_ratio'] = v
            elif k == 'h0':
                self.model_params['hubble'] = v * 100.
            elif k == 'log1e10As':
                self.model_params['scalar_amp'] = np.exp(v)*1E-10
            elif k == 'n_s':
                self.model_params['scalar_spectral_index'] = v
            elif k == 'tau':
                self.model_params['re_optical_depth'] = v
            # otherwise it's a direct translation                
            else:
                self.model_params[k] = params[k]
                
    def set_mode(self, mode):
        """Set the mode of interests here, the mode should be a
        three character string with T or F corresponding to
        scaler, vector and tensor mode, default to TFF"""
        self.mode = mode

    def _update_mode(self):
        """Populate the mode information into model parameters"""
        self.set_model_params({
            "get_scalar_cls": self.mode[0],
            "get_vector_cls": self.mode[1],
            "get_tensor_cls": self.mode[2]
        })

    def run(self):
        """Run the cosmology model to get power spectra"""
        # define base parameters
        self.camb.set_base_params(self.base_params)

        # update the modes of interests (default to scaler only)
        self._update_mode()

        # populate the user defined model parameters
        for k, v in self.model_params.items():
            self.camb.ini.set(k, v)

        # clean-up the folder to avoid reading wrong files
        self.camb.cleanup()

        # run camb sessions
        self.camb.run()

        # load power spectra into memory
        self._load_ps()

        # return a user defined target spectrum
        return self.target_spectrum(self)

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

    def fisher_matrix(self, ells, cov, targets=[], ratio=0.01):
        """Calculate the fisher matrix for a given covariance matrix

        Args:
            cov: covariance matrix lx4x4
            targets: targeting cosmological parameters as a list
            ratio: percentage step size to estimate derivatives
        """
        if len(targets)==0:
            raise ValueError("No targets found, what do you want?")

        model = self.model_params

        dCldp_list = []
        for p in targets:
            if not p in model.keys():
                print("Warning: %s not found in model, skipping..." % p)
                continue

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

            self.set_model_params(new_model_m2)
            ps_m2 = self.run()

            self.set_model_params(new_model_m1)
            ps_m1 = self.run()

            self.set_model_params(new_model_p1)
            ps_p1 = self.run()

            self.set_model_params(new_model_p2)
            ps_p2 = self.run()

            # calculate differenciations to p
            dCldp = (4.0/3.0*(ps_p1[:,1:]-ps_m1[:,1:]) - 1.0/6.0*(ps_p2[:,1:]-ps_m2[:,1:]))/h

            # interpolate it into the given ells
            dCldp_interp = interp1d(ps_m2[:,0], dCldp.T)(ells).T

            # store it to a list
            f = 2*np.pi/(ells*(ells+1))[:, None]
            dCldp_list.append(dCldp_interp*f)

        n_params = len(dCldp_list)
        alpha = np.zeros([n_params, n_params])
        n_ell = cov.shape[0]
        for i in range(n_params):
            for j in range(n_params):
                for l in np.arange(2, n_ell):
                    alpha[i,j] += np.einsum('i,ij,j', dCldp_list[i][l, :], np.linalg.inv(cov[l,:,:]), dCldp_list[j][l, :])

        return alpha

"""This script contains wrapper class that interacts with all
components of the library to perform analysis such as fisher matrix
estimation, etc.
"""

from .cambex import CambSession


class Cosmology(object):
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
        self.model_params.update(params)

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
            
        # run camb sessions
        self.camb.run()
        
        # load power spectra into memory
        self._load_ps()
        
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

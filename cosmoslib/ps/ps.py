"""Reusable functions related to camb and power spectrum calculation

This module collects some reusable functions that I used when working
with camb, power spectrum and covariance matrix
"""

import numpy as np
import scipy


class PS:
    """A container for CMB power spectrum"""
    def __init__(self, arg=None, order=('ell','TT','EE','BB','TE'), prefactor=True):
        """Simple power spectrum data wrapper

        Args:
            arg (str or ndarray): input data, can be a string to a file to load or
                an np.ndarray that contains the power spectrum. The array has to have
                a shape like [n_ell, n_spec].
            order (tuple(str)): order of columns in the input ps. Follow the naming
                convention like ell,TT,EE,BB,TE which is default
            prefactor (bool): whether input array has l(l+1)/2\pi prefactor included
        """
        # we will store power spectrum in a dictionary
        self.ps = {}
        self.prefactor=prefactor
        self.order=order
        # populate ps depending on the inputs
        if type(arg) == str:
            self.load_file(arg, order, prefactor)
        elif type(arg) == np.ndarray:
            self.load_arr(arg, order, prefactor)

    def load_arr(self, arr, order=('ell','TT','EE','BB','TE'), prefactor=True):
        """Load data from a given array"""
        if arg.shape[-1] != len(order):
            raise ValueError("provided order doesn't match the input array!")
        for i,c in enumerate(self.order):
            self.ps[c] = arr[:,i]
        # now populate fields
        self.prefactor = prefactor
        self.order = order
        self._parse_info()

    def load_file(self, infile, order=('ell','TT','EE','BB','TE'), prefactor=True):
        """load ps from a given file, will be read using np.readtxt"""
        data = np.loadtxt(infile)
        self.load_arr(data, order, prefactor)

    def __add__(self, other):
        if type(other) != PS:
            raise NotImplementedError("Currently only support PS type ops!")
        # check for ell mismatch
        if np.any(self.ell != other.ell):
            print("Warning: ells mismatch, interpolating...")
            return self.resample(other.ell) + other.resample(self.ell)
        # find common specs
        new_order = ['ell'] + [s for s in self.specs if s in other.specs]
        if len(new_order) < 2: raise ValueError("No common specs!")
        if self.prefactor != other.prefactor:
            # if prefactor mismatch, add prefactor to both of them
            self.add_prefactor()
            other.add_prefactor()
        new_ps = PS(order=new_order)
        for s in new_ps.specs:
            new_ps.ps[s] = self.ps[s] + other.ps[s]
        return new_ps

    def __sub__(self, other):
        if type(other) != PS:
            raise NotImplementedError("Currently only support PS type ops!")
        # check for ell mismatch
        if np.any(self.ell != other.ell):
            raise ValueError("ell mismatch!")
        # find common specs
        new_order = ['ell'] + [s for s in self.specs if s in other.specs]
        if len(new_order) < 2: raise ValueError("No common specs!")
        if self.prefactor != other.prefactor:
            # if prefactor mismatch, add prefactor to both of them
            self.add_prefactor()
            other.add_prefactor()
        new_ps = PS(order=new_order)
        for s in new_ps.specs:
            new_ps.ps[s] = self.ps[s] - other.ps[s]
        return new_ps

    @classmethod
    def from_arr(cls, arr, order=('ell','TT','EE','BB','TE'), prefactor=True):
        return cls(arr, order, prefactor)

    @property
    def lmin(self):
        return self.ps['ell'].min()

    @property
    def lmax(self):
        return self.ps['ell'].max()

    @property
    def ell(self):
        return self.ps['ell']

    @property
    def specs(self):
        return [o for o in self.order if o != 'ell']

    @property
    def values(self):
        # made sure ell starts from index 0
        return np.vstack([self.ps[s] for s in self.order])

    def add_prefactor(self, inplace=True):
        if self.prefactor: return self
        if inplace:
            ell = self.ell
            for c in self.specs:
                self.ps[c] *= (ell+1)*ell/(2*np.pi)
            self.prefactor = False
            return self
        else:
            return PS(self.values,self.order,prefactor=False).add_prefactor()

    def remove_prefactor(self, inplace=True):
        if not self.prefactor: return self
        if inplace:
            ell = self.ell
            for c in self.specs:
                self.ps[c] *= 2*np.pi/(ell*(ell+1))
            self.prefactor = True
            return self
        else:
            return PS(self.values,self.order,prefactor=True).remove_refactor()

    def resample(self, new_ell):
        ell = self.ell
        # make sure we are within interpolation range
        m = np.logical_and(new_ell<=self.lmax,new_ell>=self.lmin)
        # create a new ps object
        new_ps = PS(order=self.order)
        for s in self.specs:
            new_ps.ps[s] = scipy.interpolate.interp1d(ell,self.ps[s])(new_ell[m])
        return new_ps

    def plot(self, fmt="-", name='C_\ell', axes=None, ncol=2,
             legend=True, legend_below=True, filename=None,
             prefactor=True, **kwargs):
        """Plot the power spectra"""
        ell = self.ell
        if prefactor:
            self.add_prefactor()

        if not np.any(axes):
            fig, axes = plt.subplots(2,2,figsize=(12,9))

        for i,s in enumerate(self.specs):
            spec = self.ps[s]
            ax = axes[i//ncol,i%ncol]
            if self.prefactor:
                spec_name = r'$\ell(\ell+1)%s^{\rm %s}/2\pi$' % (name, s)
            else:
                spec_name = r'$%s^{\rm %s}$' % (name, s)
            if np.any(self.ps[s] < 0):
                spec = np.abs(spec)
            ax.loglog(ell, spec, fmt, **kwargs)
            ax.set_xlabel(r'$\ell$')
            ax.set_ylabel(spec_name)
            if legend and not legend_below:
                ax.legend()
        plt.tight_layout()
        if legend and legend_below:
            ax.legend(ncol=4, bbox_to_anchor=(0.6, -0.2), frameon=False)
        if filename:
            plt.savefig(filename, bbox_inches='tight')
        return axes

    def gen_realization(self):
        """Generate a realization of the power spectra"""
        # make sure we have everything we want
        target = ['ell','TT','EE','BB','TE']
        ok = [s for s in target if s in self.order] == target
        if not ok:
            raise ValueError("PS does not contain all of ell,TT,EE,BB,TE required")
        data = np.hstack([self.ps[s].reshape(-1,1) for s in target])
        rdata = gen_ps_realization(data, self.prefactor)
        new_ps = PS(order=target, prefactor=self.prefactor)
        for i,s in enumerate(target): new_ps.ps[i] = rdata[:,i]
        return new_ps


class SimpleNoise(PS):
    def __init__(self, noise, fwhm, lmax, prefactor=True):
        self.order = ('ell','TT','EE','BB','TE')
        self.prefactor = prefactor
        ell = np.arange(lmax+1)
        NlTT = noise**2*np.exp(ell*(ell+1)*beam_size**2/(8.*np.log(2)))
        NlPP = 2*NlTT
        self.ps = {'ell': ell, 'TT': NlTT, 'EE': NlPP,
                   'BB': NlPP, 'TE': np.zeros_like(ell)}


def _check_ps(ps):
    """Check the type of power spectra"""
    # if ps is a 2D array
    if len(ps.shape)>1:
        # if ps has five columns -> tensor-like
        if ps.shape[-1] == 5:
            return "TENSOR"
        # if ps has four columns -> scaler-like
        elif ps.shape[-1] == 4:
            return "SCALER"
        # not sure what's inside
        else:
            return None
    else:
        raise ValueError

def add_prefactor(ps):
    """Add the l(l+1)/2\pi prefactor in a power spectrum"""
    # check the dimension of power spectra
    ells = ps[:, 0]
    for i in range(1,ps.shape[1]):
        ps[:,i] /= 2*np.pi/(ells*(ells+1))
    return ps

def remove_prefactor(ps):
    """Remove the l(l+1)/2\pi prefactor in a power spectrum"""
    ells = ps[:, 0]
    for i in range(1,ps.shape[1]):
        ps[:,i] *= 2*np.pi/(ells*(ells+1))
    return ps

def resample(ps, ell):
    ell_old = ps[:, 0]

    # interpolate into the theory,
    tt_old = ps[:, 1]
    ee_old = ps[:, 2]
    bb_old = ps[:, 3]
    te_old = ps[:, 4]

    tt_predicted = scipy.interpolate.interp1d(ell_old, tt_old)(ell)
    te_predicted = scipy.interpolate.interp1d(ell_old, te_old)(ell)
    ee_predicted = scipy.interpolate.interp1d(ell_old, ee_old)(ell)
    bb_predicted = scipy.interpolate.interp1d(ell_old, bb_old)(ell)

    cl_predicted = np.stack([ell, tt_predicted, ee_predicted, bb_predicted, te_predicted], axis=1)

    return cl_predicted


def N_l(ells, power_noise, beam_size, prefactor=True):
    """Calculate the noise spectra for a given noise-level and beam size.

    Args:
        ells: 1D numpy array of ells
        power_noise: noise per pixel in muK-rad
        beam_size: beam size (FWHM) in unit of rad

    Returns:
        ps: len(ells) x 3 array, first column is ells
            the second column is N_lTT the third column
            is N_lPP.
    """
    NlT = power_noise**2*np.exp(ells*(ells+1)*beam_size**2/(8.*np.log(2)))
    NlP = 2 * NlT

    Nls = np.stack([ells, NlT, NlP, NlP, np.zeros(NlP.shape)], axis=1)
    if prefactor:
        Cl2Dl_(Nls)
    return Nls

def add_noise_nl(ps, power_noise, beam_size, l_min, l_max, prefactor=True):
    """Add the noise term Nl to the power spectra based on the telescope noise
    properties.

    Args:
        ps: power spectra
        power_noise: power noise in \muK rad
        beam_size: FWHM angular resolution of the beam in rad
        l_min, l_max: limits of ells

    Returns:
        ps: power spectra with noises Nl added
    """
    ells = ps[:,0]

    # calculate noise spectra
    Nls = N_l(ells, power_noise, beam_size, prefactor=False)

    new_ps = ps.copy()
    if prefactor:
        new_ps = remove_prefactor(new_ps)

    NlT = Nls[:,1]
    NlP = Nls[:,2]
    new_ps[:,1] += NlT
    new_ps[:,2] += NlP
    new_ps[:,3] += NlP

    if prefactor:
        new_ps = add_prefactor(new_ps)

    mask = np.logical_and(ells>=l_min, ells<=l_max)

    return new_ps[mask,:]


def Dl2Cl(ps, inplace=False):
    """Add the l(l+1)/2\pi prefactor in a power spectrum"""
    # check the dimension of power spectra
    ells = ps[:, 0]
    if not inplace:
        new_ps = ps.copy()
    else:
        new_ps = ps
    for i in range(1,ps.shape[1]):
        new_ps[:,i] /= 2*np.pi/(ells*(ells+1))
    return new_ps

def Cl2Dl(ps, inplace=False):
    """Remove the l(l+1)/2\pi prefactor in a power spectrum"""
    ells = ps[:, 0]
    if not inplace:
        new_ps = ps.copy()
    else:
        new_ps = ps
    for i in range(1,ps.shape[1]):
        new_ps[:,i] *= 2*np.pi/(ells*(ells+1))
    return new_ps


def gen_ps_realization(ps, prefactor=True):
    """Generate a random power spectra realization

    Args:
        ps: power spectra
        prefactor: true if ps is Dl

    Returns:
        ps realization: consistent with prefactor choice
    """
    # first make a copy to make sure we don't affect the original
    if prefactor:
        ps = Dl2Cl(ps, inplace=False)
    ells, ClTT, ClEE, ClBB, ClTE = ps[:,0], ps[:,1], ps[:,2], ps[:,3], ps[:,4]

    # define empty arrays to hold the generated power spectra
    m_ps = np.zeros_like(ps)
    m_ps[:,0] = ells

    # this is certainly slow, but i don't need to run this very often
    # so it's fine to leave it like this. in principle i don't need to
    # keep all the negative part as well. These can be improved if
    # performance becomes an issue
    for i in range(len(ells)):
        l = int(ells[i])

        # generate gaussian random complex numbers with unit variance
        zeta1 = np.random.randn(l+1)*np.exp(1j*np.random.rand(l+1)*2*np.pi)
        zeta2 = np.random.randn(l+1)*np.exp(1j*np.random.rand(l+1)*2*np.pi)
        zeta3 = np.random.randn(l+1)*np.exp(1j*np.random.rand(l+1)*2*np.pi)

        # for m=0, zeta has to be real
        zeta1[0] = np.abs(zeta1[0])
        zeta2[0] = np.abs(zeta2[0])
        zeta3[0] = np.abs(zeta3[0])

        # generate alm
        aTlm = zeta1 * ClTT[i]**0.5
        aElm = zeta1 * ClTE[i] / (ClTT[i])**0.5 + zeta2*(ClEE[i] - ClTE[i]**2/ClTT[i])**0.5
        aBlm = zeta3 * ClBB[i]**0.5

        i_ClTT = (aTlm[0]**2 + 2*(np.sum(np.abs(aTlm[1:])**2)))/(2*l+1)
        i_ClEE = (aElm[0]**2 + 2*(np.sum(np.abs(aElm[1:])**2)))/(2*l+1)
        i_ClBB = (aBlm[0]**2 + 2*(np.sum(np.abs(aBlm[1:])**2)))/(2*l+1)
        i_ClTE = (aTlm[0]*aElm[0] + 2*(np.sum(np.conj(aTlm[1:])*aElm[1:])))/(2*l+1)

        # assign the new values to the new array
        m_ps[i,1] = i_ClTT
        m_ps[i,2] = i_ClEE
        m_ps[i,3] = i_ClBB
        m_ps[i,4] = i_ClTE

    if prefactor:
        return Cl2Dl(m_ps, inplace=True)
    else:
        return m_ps


def covmat(ps, pixel_noise, beam_size, l_min,
           l_max, f_sky, prefactor=True):
    """Calculate the covariance matrix based on a model.

    Args:
        ps: power spectra
        pixel_noise: noise per pixel
        beam_size: beam size in degress (full width half minimum)
        l_min, l_max: range of ells
        f_sky: sky coverage fraction, 1 means full-sky coverage

    Returns:
        cov: a tensor of size [n_ell, n_ps, n_ps], for example with
             a lmax of 5000, the tensor size will be [5000, 4, 4]
    """
    # assuming the beam is a gaussian beam with an ell dependent
    # beam size
    if prefactor:
        remove_prefactor(ps)

    _ells = ps[:, 0]

    Wb = lambda l: np.exp(l*(l+1)*beam_size**2/(8.*np.log(2)))

    # calculate the noise parameter w^-1
    wTinv = pixel_noise**2
    wPinv = 2*wTinv

    mask = np.logical_and(_ells>=l_min, _ells<=l_max)

    # extract power spectra
    ells = ps[mask,0]
    ClTT = ps[mask,1]
    ClEE = ps[mask,2]
    ClBB = ps[mask,3]
    ClTE = ps[mask,4]

    # initialize empty covariance tensor. Since the covariance matrix
    # depends on ell, we will make a higher dimensional array [n_ell,
    # n_ps, n_ps] where the first index represents different ells, the
    # second and third parameters represents different power spectra
    n_ells = len(ells)
    cov = np.zeros([n_ells, 4, 4])

    for (i, l) in enumerate(ells):
        # T, T
        cov[i,0,0] = 2.0/(2*l+1)*(ClTT[i] + wTinv*Wb(l))**2

        # E, E
        cov[i,1,1] = 2.0/(2*l+1)*(ClEE[i] + wPinv*Wb(l))**2

        # B, B
        cov[i,2,2] = 2.0/(2*l+1)*(ClBB[i] + wPinv*Wb(l))**2

        # TE, TE
        cov[i,3,3] = 1.0/(2*l+1)*(ClTE[i]**2 + (ClTT[i] + wTinv*Wb(l))
                                  *(ClEE[i] + wPinv*Wb(l)))

        # T, E
        cov[i,0,1] = cov[i,1,0] = 2.0/(2*l+1)*ClTE[i]**2

        # T, TE
        cov[i,0,3] = cov[i,3,0] = 2.0/(2*l+1)*ClTE[i]*(ClTT[i] +
                                                       wTinv*Wb(l))

        # E, TE
        cov[i,1,3] = cov[i,3,1] = 2.0/(2*l+1)*ClTE[i]*(ClEE[i] +
                                                       wPinv*Wb(l))

    # now we include the effect of partial sky coverage
    cov /= f_sky

    if prefactor:
        add_prefactor(ps)

    return ells, cov


def fisher_matrix(model, cov, ratio=0.01):
    """Estimate the fisher matrix near the model provided.

    Args:
        model: a python dictionary that contains the cosmological
        parameters that one is interested in estimating the fisher
        matrix between them (fiducial model)

        cov: the covariance matrix between the power spectra

        ratio: the ratio of the parameter to vary. This is because fisher
        matrix calculation involves differenciating the power spectra
        with respect to the parameters given. To estimate the derivatives
        one needs a small variation near the best fit value. This ratio
        characterize how much variation is needed.

    Returns:
        alpha: fisher matrix
        params: a list of the parameters corresponding to each index
    """
    # first we need to generate multiple models near the best fit model
    # we will generate four models for each parameter, so in total the
    # underlying CAMB is called n_par * 4 times. First we create an empty
    # list to hold our derivatives. Each element in dCldp will correspond
    # to the derivative of the corresponding parameter in parameters list
    params = [key for key in model.keys()]
    n_params = len(params)
    dCldp = []

    for p in params:
        # generate four models by tweaking parameters
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

        # generate four sets of parameters
        pars_m1 = generate_camb_params(**new_model_m1)
        pars_m2 = generate_camb_params(**new_model_m2)
        pars_p1 = generate_camb_params(**new_model_p1)
        pars_p2 = generate_camb_params(**new_model_p2)

        # get power spectra for both models
        ClTT_m1, ClEE_m1, ClBB_m1, ClTE_m1 = generate_cmb_power_spectra(pars_m1)
        ClTT_m2, ClEE_m2, ClBB_m2, ClTE_m2 = generate_cmb_power_spectra(pars_m2)
        ClTT_p1, ClEE_p1, ClBB_p1, ClTE_p1 = generate_cmb_power_spectra(pars_p1)
        ClTT_p2, ClEE_p2, ClBB_p2, ClTE_p2 = generate_cmb_power_spectra(pars_p2)

        # calculate differenciations to p
        dClTTdp = (4.0/3.0 * (ClTT_p1-ClTT_m1) -1.0/6.0 * (ClTT_p2 - ClTT_m2)) / h
        dClEEdp = (4.0/3.0 * (ClEE_p1-ClEE_m1) -1.0/6.0 * (ClEE_p2 - ClEE_m2)) / h
        dClBBdp = (4.0/3.0 * (ClBB_p1-ClBB_m1) -1.0/6.0 * (ClBB_p2 - ClBB_m2)) / h
        dClTEdp = (4.0/3.0 * (ClTE_p1-ClTE_m1) -1.0/6.0 * (ClTE_p2 - ClTE_m2)) / h

        dCldp.append(np.vstack([dClTTdp, dClEEdp, dClBBdp, dClTEdp]))

    # now that the derivatives are calculated, we are ready to calculate the
    # fisher matrix alpha.
    # TODO: This calculation can perhaps be optimized
    alpha = np.zeros([n_params, n_params])
    n_ell = cov.shape[0]
    for i in range(n_params):
        for j in range(n_params):
            for l in np.arange(n_ell):
                alpha[i,j] += np.einsum('i,ij,j', dCldp[i][:,l], np.linalg.inv(cov[l,:,:]), dCldp[j][:,l])

    return alpha, params


# utility functions
def substract_ps(ps1, ps2):
    if ps1.shape[-1] != ps2.shape[-1]:
        raise "Power spectra shape mismatch"
    else:
        ps1_lmin = ps1[0][0]
        ps2_lmin = ps2[0][0]
        lmin = max(ps1_lmin, ps2_lmin)
        ps1_lmax = ps1[-1][0]
        ps2_lmax = ps2[-1][0]
        lmax = min(ps1_lmax, ps2_lmax)
        ps1_mask = np.logical_and(ps1[:,0] >= lmin, ps1[:,0] <= lmax)
        ps1_match = ps1[ps1_mask,:]
        ps2_mask = np.logical_and(ps2[:,0] >= lmin, ps2[:,0] <= lmax)
        ps2_match = ps2[ps2_mask,:]
        ps1_match[:,1:] -= ps2_match[:,1:]
        return ps1_match

def divide_ps(ps1, ps2):
    if ps1.shape[-1] != ps2.shape[-1]:
        raise "Power spectra shape mismatch"
    else:
        ps1_lmin = ps1[0][0]
        ps2_lmin = ps2[0][0]
        lmin = max(ps1_lmin, ps2_lmin)
        ps1_lmax = ps1[-1][0]
        ps2_lmax = ps2[-1][0]
        lmax = min(ps1_lmax, ps2_lmax)
        ps1_mask = np.logical_and(ps1[:,0] >= lmin, ps1[:,0] <= lmax)
        ps1_match = ps1[ps1_mask,:]
        ps2_mask = np.logical_and(ps2[:,0] >= lmin, ps2[:,0] <= lmax)
        ps2_match = ps2[ps2_mask,:]
        ps1_match[:,1:] /= ps2_match[:,1:]
        return ps1_match

def add_ps(ps1, ps2):
    if ps1.shape[-1] != ps2.shape[-1]:
        raise "Power spectra shape mismatch"
    else:
        ps1_lmin = ps1[0][0]
        ps2_lmin = ps2[0][0]
        lmin = max(ps1_lmin, ps2_lmin)
        ps1_lmax = ps1[-1][0]
        ps2_lmax = ps2[-1][0]
        lmax = min(ps1_lmax, ps2_lmax)
        ps1_mask = np.logical_and(ps1[:,0] >= lmin, ps1[:,0] <= lmax)
        ps1_match = ps1[ps1_mask,:]
        ps2_mask = np.logical_and(ps2[:,0] >= lmin, ps2[:,0] <= lmax)
        ps2_match = ps2[ps2_mask,:]
        ps1_match[:,1:] += ps2_match[:,1:]
        return ps1_match

def multiply_ps(ps1, ps2):
    if ps1.shape[-1] != ps2.shape[-1]:
        raise "Power spectra shape mismatch"
    else:
        ps1_lmin = ps1[0][0]
        ps2_lmin = ps2[0][0]
        lmin = max(ps1_lmin, ps2_lmin)
        ps1_lmax = ps1[-1][0]
        ps2_lmax = ps2[-1][0]
        lmax = min(ps1_lmax, ps2_lmax)
        ps1_mask = np.logical_and(ps1[:,0] >= lmin, ps1[:,0] <= lmax)
        ps1_match = ps1[ps1_mask,:]
        ps2_mask = np.logical_and(ps2[:,0] >= lmin, ps2[:,0] <= lmax)
        ps2_match = ps2[ps2_mask,:]
        ps1_match[:,1:] *= ps2_match[:,1:]
        return ps1_match

def power_ps(ps, n):
    ps_new = ps.copy()
    ps_new[:,1:] = ps_new[:,1:]**n
    return ps_new

def multiply_ps_num(ps, n):
    ps_new = ps.copy()
    ps_new[:,1:] = ps_new[:,1:]*n
    return ps_new

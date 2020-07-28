"""Calculation related to magnetic fields

"""
import numpy as np
import scipy.special as sp
from scipy.special import spherical_jn
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from tqdm import tqdm

#############
# constants #
#############

e = 1.602e-19  # C

###############################
# transfer function for cl_aa #
###############################

class MagneticField:
    def __init__(self, P_k=None):
        """Calculate various magnetic field related quantities here

        Parameters
        ----------
        P_k: 2-pt correlation function of magnetic field (vectorized func of k)

        """
        self.P_k = P_k

    def delta_m2(self, k, lam0):
        """Calculate \Delta_M^2 following Eq. (12)

        Parameters
        ----------
        k: wavenumbers of interests
        lam0: observing wavelength

        """
        return k**3*self.P_k(k)*(3*lam0**2/(16*np.pi**2*e))**2

    def set_pk(self, B_lambda, n_B):
        """Set the primordial magnetic field power spectrum based on an
        amplitude and spectral index

        Eq. (100) from arxiv 0911.2714

        """
        k_lambda = 1  # FIXME: placeholder
        A = (2*np.pi)*(n_B+5)*B_lambda**2 / (sp.gamma((n_B+3)/2)*k_lambda**(n_B+3))
        self.P_k = lambda k: A*k**n_B


class ClaaTransferFunction:
    def __init__(self, cosmo=None, mag=None, verbose=True):
        """Calculate the transfer function for Cl_\alpha\alpha from given
        primordial magnetic field power spectrum following arxiv 1106.1438.

        Parameters
        ----------
        cosmo: CAMBdata object containing the background cosmology
        mag: MagneticField object
        verbose: option to be verbose

        """
        self.cosmo = cosmo
        self.mag = mag
        self.verbose = verbose

    def T(self, lmax, k, n_eta=1000, max_mem=100):
        """In camb, opacity -> a n_e \sigma_T

        T_L(k) = \int d\eta a n_e \sigma_T j_L(k(\eta_0-\eta))  -- Eq. (30)
        T_L^1(k) = 1/(2L+1) [L T_{L-1}(k) - (L+1)T_{L+1}]       -- Eq. (33)

        Parameters
        ----------
        lmax: maximum ell to compute upto
        k: value of wavenumber to evaluate at
        n_eta: number of values of etas to evaluate to use as spline
        max_mem: maximum memory allowed to use in Gb

        Returns
        -------
        (T[n_ell, n_k], T1[n_ell, n_k]) with n_ell = lmax+1

        """
        # calculate the amount of memory roughly and split ells into
        # multiple parts so we don't exceed the maximum memory specified
        mem = (lmax+2)*n_eta*len(k)*8/1024**3 * 1.2
        if self.verbose: print(f"-> Estimate memory: {mem:.1f}Gb")
        if self.verbose: print(f"-> Max allowed memory: {max_mem:.1f}Gb")
        nparts = int(np.ceil(mem / max_mem))
        # convention is everything with _ is a flat version
        ells_ = np.arange(lmax+2)
        if nparts > 1: ells_parts = np.array_split(ells_, nparts)
        else: ells_parts = [ells_]
        if self.verbose: print(f"-> Split ells into: {nparts:d} parts")
        # get eta ranges to evalute based on given cosmology
        eta_0 = self.cosmo.tau0  # comoving time today
        eta_star = self.cosmo.tau_maxvis  # comoving time at maximum of visibility
        etas_ = np.linspace(eta_star, eta_0, n_eta)  # flat array
        k_ = k  # alias to be consistent
        Tlks_list = []  # to store Tlk for each part
        ells_list = []
        # define a combined operation of splined integration using
        spline_int = lambda x: quad(CubicSpline(etas_, x), eta_star, eta_0)[0]
        for i in tqdm(range(len(ells_parts))):
            ells_ = ells_parts[i]
            # allocate each array to a seperate axis so broadcasting works properly
            ells, k, etas = np.ix_(ells_, k_, etas_)
            # don't need to do this every time but we don't expect to run lots of parts
            dtau = self.cosmo.get_background_time_evolution(etas, vars=['opacity'], format='array')
            jlk = spherical_jn(ells, k*(eta_0-etas))
            integrand = dtau * jlk
            del dtau, jlk
            # apply spline integration to the right axis
            Tlk = np.apply_along_axis(spline_int, -1, integrand)
            del integrand
            # store results for each part
            Tlks_list.append(Tlk)
            ells_list.append(ells)
        # stack Tlk from different parts
        if self.verbose: print(f"-> Stacking {nparts:d} parts")
        Tlk = np.vstack(Tlks_list)
        ells = np.vstack(ells_list)
        del Tlks_list, ells_list
        # calculate Tl1k
        Tl1k = np.zeros_like(Tlk)
        Tlkp1 = Tlk[2:,...]   # l=2, lmax+1
        Tlkm1 = Tlk[:-2,...]  # l=0, lmax-1
        ells = ells[1:-1].reshape(-1,1)  # l=1, lmax
        # calculate T1lk for l=1, lmax
        Tl1k[1:-1,...] = 1/(2*ells+1)*(ells*Tlkm1 - (ells+1)*Tlkp1)
        # calculate T1lk for l=0
        Tl1k[0,...] = -Tlk[1,...]  # Eq 10.51.2
        del ells, Tlkp1, Tlkm1
        # return up to lmax
        return Tlk[:-1,...], Tl1k[:-1,...]

    def claa(self, lmin, lmax, kmin, kmax, lam0, nk=1000, n_eta=1000, max_mem=100):
        """Calculate C_l^{\alpha\alpha} following Eq. (31)

        Parameters
        ----------
        lmin, lmax: range of ells of interests
        kmin, kmax: range of wavenumber k to integrate
        lam0: observing wavelength
        nk: number of k to compute to use as a basis for interpolation

        """
        # get transfer functions
        assert lmin>0, "Only support lmin>0 at the moment"
        logk_ = np.linspace(np.log(kmin), np.log(kmax), nk)
        # if self.Tlk is None or self.T1lk is None:
        Tlk, T1lk = self.T(lmax+1, np.exp(logk_), n_eta, max_mem)
        ells_ = np.arange(lmin, lmax+1, dtype=int)
        Tm1 = Tlk[ells_-1, :]
        Tp1 = Tlk[ells_+1, :]
        T1 = T1lk[ells_, :]
        print(np.sum(Tlk, axis=1))
        del Tlk, T1lk
        # make sure ells are aligned with the axis it's supposed to broadcast to
        ells, logk = np.ix_(ells_, logk_)
        integrand = (ells/(2*ells+1)*Tm1**2 + (ells+1)/(2*ells+1)*Tp1**2 - T1**2)
        integrand *= self.mag.delta_m2(np.exp(logk), lam0)
        del Tm1, Tp1, T1
        # make a spline interpolator to be used for integration from logk to integrand
        spline_int = lambda x: quad(CubicSpline(logk_, x), np.log(kmin), np.log(kmax))[0]
        claa = 2/np.pi*np.apply_along_axis(spline_int, -1, integrand)
        del integrand
        return ells_, claa.ravel()

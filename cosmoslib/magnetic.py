"""Calculation related to magnetic fields

"""
import numpy as np
from scipy.special import spherical_jn
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from tqdm import tqdm


###############################
# transfer function for cl_aa #
###############################

class ClaaTransferFunction:
    def __init__(self, cosmo, verbose=True):
        """Calculate the transfer function for Cl_\alpha\alpha from given
        primordial magnetic field power spectrum following arxiv 1106.1438.

        Parameters
        ----------
        cosmo: CAMBdata object containing the background cosmology

        """
        self.cosmo = cosmo
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
            if self.verbose: print(f"-> Work on part {i}")
            ells = ells_parts[i]
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

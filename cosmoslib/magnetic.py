"""Calculation related to magnetic fields

"""
import numpy as np
import scipy.special as sp
from scipy.special import spherical_jn
from scipy.interpolate import CubicSpline
from scipy.integrate import quad, romberg
from tqdm import tqdm
from cosmoslib.units import natural as u

###############################
# transfer function for cl_aa #
###############################

class MagneticField:
    def __init__(self, B_lambda, n_B, lam=1, h=0.673):
        """Calculate various magnetic field related quantities here

        Parameters
        ----------
        B_lambda: amplitude of PMF after smoothing in nG
        P_k: 2-pt correlation function of magnetic field (vectorized func of k)
        h: reduced hubble's parameter in unit of 100km/s/Mpc
        """
        self.B_lambda = B_lambda
        self.n_B = n_B
        self.lam = lam
        self.h = h
        # compute amplitude
        # Eq. (100) from arxiv 0911.2714. With an additional factor of 0.5 to match
        # the convention used in other papers such as Kosowsky (2005) and Pagosian (2013)
        # k_lambda = 2*np.pi / lam
        # self.A = (2*np.pi)**(n_B+5)*(B_lambda*u.nG)**2 / (2*sp.gamma((n_B+3)/2)*k_lambda**(n_B+3))
        # same expression but simplified
        self.A = (2*np.pi)**2*(B_lambda*u.nG)**2 / (2*sp.gamma((n_B+3)/2)) * lam**(n_B+3)

    def delta_m2(self, k, freq):
        """Calculate \Delta_M^2 following Eq. (12)

        Parameters
        ----------
        k: wavenumbers of interests
        freq: frequency in GHz
        """
        v0 = freq * u.GHz
        return k**3*self.P_k(k)*(3/((16*np.pi**2*u.e)*v0**2))**2

    def P_k(self, k):
        """find the primordial magnetic field power spectrum based on an
        amplitude and spectral index
        Parameters
        ----------
        k (np.ndarray): wavenumbers of interests

        """
        kD = self.kDissip()
        Pk = np.zeros_like(k, dtype=np.double)
        Pk[k<kD] = self.A*k**self.n_B
        return Pk

    def kDissip(self):
        """
        Returns
        -------
        kD: in unit of Mpc^-1

        """
        k_lambda = 2*np.pi / self.lam
        # version from Planck 2015
        kD = 5.5e4 * self.B_lambda**(-2) * k_lambda**(self.n_B+3) * self.h
        # version from Kosowsky 2005
        # kD = 2.9e4 * self.B_lambda**(-2) * k_lambda**(self.n_B+3) * self.h
        kD = kD**(1/(self.n_B+5))
        return kD


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

    def T(self, lmax, k, n_eta=1000):
        """In camb, opacity -> a n_e \sigma_T

        T_L(k) = \int d\eta a n_e \sigma_T j_L(k(\eta_0-\eta))  -- Eq. (30)

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
        if self.verbose: print(f"-> Eta integrated from {eta_0:.1f} to {eta_star:.1f}")
        k_ = k  # alias to be consistent
        Tlks_list = []  # to store Tlk for each part
        ells_list = []
        # define a combined operation of splined integration using
        spline_int = lambda x: quad(CubicSpline(etas_, x), eta_star, eta_0, epsrel=1e-4)[0]
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
        ells = ells[1:-1].reshape(-1,1)  # l=1, lmax
        # return up to lmax
        return ells, Tlk[:-1,...]

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
        spline_int = lambda x: romberg(CubicSpline(logk_, x), np.log(kmin), np.log(kmax))
        claa = 2/np.pi*np.apply_along_axis(spline_int, -1, integrand)
        del integrand
        return ells_, claa.ravel()


def jn_first_zero(n):
    """Get an approximated location for the first zero of
    spherical bessel's function at a given order n"""
    precomputed = [3.14159, 4.49341, 5.76346, 6.98793, 8.18256, 9.35581, 10.5128,
                   11.657, 12.7908, 13.9158, 15.0335, 16.1447, 17.2505, 18.3513,
                   19.4477, 20.5402, 21.6292, 22.715, 23.7978, 24.878, 25.9557,
                   27.0311, 28.1043, 29.1756, 30.245, 31.3127]
    try: return precomputed[n]
    except:
        # formula 9.5.14 in Handbook of Mathematical Functions
        v = n + 0.5
        return v + 1.8557571*v**(1/3) + 1.033150*v**(-1/3) - \
            0.00397*v**(-1) - 0.0908*v**(-5/3) + 0.043*v**(-7/3)


class KosowskyClaa:
    def __init__(self, lmax, cosmo, mag):
        """Calculate Cl^aa using approximation formula in Kosowsky (2005)

        Parameters
        ----------
        lmax: maximum ell to calculate
        cosmo: camb cosmology class
        mag: magnetic field class
        """
        self.lmax = lmax
        self.cosmo = cosmo
        self.mag = mag

    def claa(self, nx1=1000, nx2=1000, freq=100, spl=CubicSpline):
        """
        Parameters
        ----------
        freq: frequency of interests in GHz
        """
        v_0 = freq * u.GHz
        eta_0 = self.cosmo.tau0  # comoving time today
        eta_maxvis = self.cosmo.tau_maxvis  # comoving time today
        # kD = self.mag.kDissip()
        kD = 2  # following Kosowsky 2005
        xd = kD
        ells = np.arange(0, self.lmax+1)
        clas = np.zeros_like(ells, dtype=np.double)
        # perform exact calculate before x = x_approx
        for i in tqdm(range(len(ells))):
            l = ells[i]
            x_approx = min(jn_first_zero(l),xd)  # approximation
            # x_approx = xd  # approximation
            x = np.linspace(0, x_approx, nx1)[1:]
            integrand = x**self.mag.n_B
            integrand *= spherical_jn(l, x)**2
            # start to make approximation after x = x_approx with j_l^2 -> 1/(2x^2)
            clas[i] = quad(spl(x, integrand), 0, x_approx)[0]
            if xd > x_approx:
                x = np.linspace(x_approx, xd, nx2)
                integrand = x**self.mag.n_B
                integrand *= 1/(2*x**2)
                clas[i] += quad(spl(x, integrand), x_approx, xd)[0]
        # reuse some numbers in mag.A
        # alpha=e^2 convention
        clas *= 9*ells*(ells+1)/(4*(2*np.pi)**5*u.e**2) * self.mag.A / eta_0**(self.mag.n_B+3) / (v_0**4)
        # 4 pi alpha= e^2 convention
        # clas *= 9*ells*(ells+1)/(8*(np.pi)**3*u.e**2) * self.mag.A / eta_0**(self.mag.n_B+3) / (v_0**4)
        return ells, clas

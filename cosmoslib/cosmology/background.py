import numpy as np
from scipy import integrate as scipy_integrate
import sympy
from pyop import O, _

import cosmoslib.units as u

# ----------------------------------------------------------------------
# * Constants
Tcmb = 2.725  # K
pi = np.pi
pi2 = pi**2
rho_over_omega = 3*100**2/(8*pi)*u.H0_unit
T_gm_0 = Tcmb * u.Kelvin

# ----------------------------------------------------------------------
# * Basic background cosmology
#
# Photon density:
# \rho_gm(z) = \rho_gm(0)(1+z)^4

def rho_gm(z, rho_gm_0):
    return rho_gm_0 * (1+z)**4

# Dark energy:
#   \rho_\lambda(z) = \rho_\lambda(0) (1+z)^(3(1+w(z)))
# where w(z) is the equation of state of of dark energy

def rho_lam(z, rho_lam_0, w):
    """w => w(z)"""
    return rho_lam_0 * (1+z)**(3*(1+w(z)))

# Cold dark matter:
#   \rho_cdm(z) = \rho_cdm(0)(1+z)^3

def rho_cdm(z, rho_cdm_0):
    return rho_cdm_0 * (1+z)**3

# Baryon:
# \rho_b(z) = \rho_b(0)(1+z)^3

def rho_b(z, rho_b_0):
    return rho_b_0 * (1+z)**3

# Curvature:
# \rho_K(z) = \rho_K(0)(1+z)^2

def rho_k(z, rho_k_0):
    return rho_k_0 * (1+z)**2

# Neutrino:
#
# There are two contributions to neutrino density:
#
# \rho_\nu = \rho_nu (massless) + rho_nu (massive)
#
# For massless: (i.e. Piattella 2018: Eq. 3.96)
#
# \rho_nu = N_nu * (7/8) (4/11)^(4/3) \rho_gm(0) * (1+z)^4
#
# where we have used
#
# T_\nu = T_\gm (4/11)^(1/3) ~ 0.714

Tnu_from_Tgm = lambda T_gm: T_gm*(4/11)**(1/3)

def rho_nu_massless(z, Nnu_massless, rho_gm_0):
    return Nnu_massless * (7/8)*(4/11)**(4/3) * rho_gm_0 * (1+z)**4

# For massive:
# There is no analytic solution, one needs to integrate the
# distribution function:
#
#                 d^3p           \sqrt{p^2c^2 + m_nu^2c^4}
# E_nu = \int ----------- * --------------------------------------
#             (2pi\hbar)^3   exp[\sqrt{p^2c^2 + m_nu^2}/k_BT] + 1
#
# We will work in unit that \hbar = c = k_B = G = 1 (planck unit).
# This integral can be simplified to (in general) (Piallette Eq. 3.104)
#
# E = [ T^4 / 2\pi^2 ] \int_A^\inf dx [x^2\sqrt{x^2-A^2} / e^x \pm 1]
#
# where A = m / T. This applies to any particles for boson with minus
# sign and fermion with plus sign. For neutrino, we get

def T_at_redshift(z):
    """Temperature at redshift z"""
    return T_0*(1+z)

# TODO: cross check with others' calculations
# For example:
# https://github.com/marius311/Cosmology.jl/blob/master/src/Background.jl#L26
# https://github.com/cmbant/CAMB/blob/65b3f21dad9579266552f5568beb5ce6590502ac/fortran/massive_neutrinos.f90#L141
# CLASS doesn't consider massive neutrinos as they have commented that
# its impacts are small
# https://github.com/lesgourg/class_public/blob/4aaa7725d3a91dc9bda67fe8d32f61d8c4d5ecab/source/input.c#L3159
# TODO: precomputation and interpolate with log(rho) vs. log(z)
def rho_nu_massive(z, Nnu_massive, m_nu):
    T_nu = T_at_redshfit(z) // O(Tnu_from_Tgm)
    A = m_nu / T_nu
    integrand = lambda x: x**2*np.sqrt(x**2 - A**2) / (np.exp(x)+1)
    return T_nu**4 / (2*pi2) * Nnu_massive * integrate(integrand, A, np.inf)

def rho_nu(z, Nnu_massless, rho_gm_0, Nnu_massive=0, m_nu=0):
    """Total neutrino energy density"""
    # FIXME: not using the massive before cross checking
    return rho_nu_massless(z, Nnu_massless, rho_gm_0) # + \
        # rho_nu_massive(z, Nnu_massive, m_nu)

# Hubble parameter is then given by
#
# H(z)^2 = 8\pi/3 [ \rho_k(z) + \rho_gm(z) + \rho_b(z) + \rho_cdm(z) + \rho_lam(z)
#                 + \rho_nu(z) ]

class Cosmology:
    """Wrapper class for cosmology calculations"""
    def __init__(self, **params):
        self.H0 = params.get('H0')
        self.h = self.H0 / 100
        # Photon
        self.rho_gm_0  = (pi2/15)*T_gm_0**4
        self.omega_gm  = self.rho_gm_0 / rho_over_omega
        self.Omega_gm  = self.omega_gm / self.h**2
        # Baryons
        self.omega_b   = params.get('omega_b')
        self.rho_b_0   = self.omega_b * rho_over_omega
        self.Omega_b   = self.omega_b / self.h**2
        # CDM
        self.omega_cdm = params.get('omega_cdm')
        self.rho_cdm_0 = self.omega_cdm * rho_over_omega
        self.Omega_cdm = self.omega_cdm / self.h**2
        # Neutrinos
        self.Nnu_massless = params.get('Nnu_massless')
        self.Nnu_massive = params.get('Nnu_massive')
        self.m_nu = params.get('m_nu')
        if self.m_nu == 0:
            self.Nnu_massless += self.Nnu_massive
            self.Nnu_massive = 0
        self.rho_nu_0  = self.rho_nu_at_z(0)
        self.omega_nu  = self.rho_nu_0 / rho_over_omega
        self.Omega_nu  = self.omega_nu / self.h**2
        # Curvature
        self.Omega_k   = params.get('Omega_k')
        self.omega_k   = self.Omega_k * self.h**2
        self.rho_k_0   = self.omega_k * rho_over_omega
        # Dark energy
        self.w_lam     = params.get('w_lam')
        self.Omega_lam = 1-self.Omega_k-self.Omega_b-self.Omega_cdm-self.Omega_nu-self.Omega_gm
        self.omega_lam = self.Omega_lam * self.h**2
        self.rho_lam_0 = self.omega_lam * rho_over_omega

    # wrap around functional calculation we layed out earlier
    def rho_gm_at_z(self, z):
        return rho_gm(z, self.rho_gm_0)

    def rho_lam_at_z(self, z):
        """w => w(z)"""
        return rho_lam(z, self.rho_lam_0, self.w_lam)

    def rho_cdm_at_z(self, z):
        return rho_cdm(z, self.rho_cdm_0)

    def rho_b_at_z(self, z):
        return rho_b(z, self.rho_b_0)

    def rho_k_at_z(self, z):
        return rho_k(z, self.rho_k_0)

    def rho_nu_at_z(self, z):
        return rho_nu(z, self.Nnu_massless, self.rho_gm_0,
                      self.Nnu_massive, self.m_nu)

    def Hubble_at_z(self, z):
        H_at_z  = np.sqrt(8*pi/3)  # TBC
        H_at_z *= np.sqrt(self.rho_k_at_z(z) + self.rho_b_at_z(z) + \
                          self.rho_cdm_at_z(z) + self.rho_lam_at_z(z) + \
                          self.rho_gm_at_z(z) + self.rho_nu_at_z(z))
        return H_at_z

    def eta_between_z1_z2(self, z1, z2):
        """Calculate difference in conformal time between z1 and z2
        positive when z2>z1"""
        integrand = lambda z: 1 / self.Hubble_at_z(z)
        return integrate(integrand, z1, z2)

    def eta_at_z(self, z):
        """Conformal time between z to now (between -∞ and 0)"""
        return self.eta_between_z1_z2(z, 0)

    def DC_at_z(self, z):
        """Comoving distance at readshift z"""
        return -self.eta_at_z(z)

    def DA_at_z(self, z):
        """Angular diameter distance"""
        d = self.DC_at_z(z)
        # curvature
        K = -self.omega_k*(100*u.km/u.second)**2  # not needed but good
                                                  # for reference
        if K == 0:
            return d
        elif K < 0:
            return 1/np.sqrt(-K)*np.sin(d*np.sqrt(-K))
        else:
            return 1/np.sqrt(k)*np.sinh(d*np.sqrt(k))

    def DL_at_z(self, z):
        """Luminosity distance"""
        return self.DA_at_z(z) * (1+z)**2

    def drs_over_dz_at_z(self, z):
        """Derivative of comoving sound horizon at redshift z"""
        R = 3*self.rho_b_0 / (4*self.rho_gm_0*(1+z))
        return 1/self.Hubble_at_z(z)/np.sqrt(3*(1+R))

    def rs_at_z(self, z):
        """Comoving sound horizon at redshift z"""
        integrand = lambda z: self.drs_over_dz_at_z(z)
        return integrate(integrand, z, np.inf)

    def zstar_HS(self):
        """Redshift at decoupling using fitting formula from Hu & Sugiyama"""
        # give shorthand name so equation look more compact
        wb = self.omega_b
        wc = self.omega_cdm
        wnu = self.omega_nu
        # fitting formula
        return 1048*(1+0.00124*wb**(-0.738))*(1+(0.0783*wb**(-0.238)/(1+39.5*wb**0.763))*(wb+wc+wnu)**(0.560/(1+21.1*wb**1.81)))

    def theta_s_at_z(self, z):
        """Angular size of the sound horizon [rad] at redshift z"""
        return self.rs_at_z(z) / self.DA(z)

    def theta_MC(self):
        """θs at the decoupling redshift calculated from zstar_HS. This is
        like CosmoMC's "theta_mc", except CosmoMC's also uses some additional
        approximations which we don't use here.
        """
        return self.theta_s_at_z(self.zstar_HS())

    # TODO: add reionization related stuff
    # utility methods: no more physics

    def _sub_all(self, var, expr):
        """Symbolic substitution in all fields"""
        for k, v in self.__dict__.items():
            if isinstance(v, sympy.Expr):
                self.__dict__[k] = v.subs(field, expr)


def integrate(fun, lo, hi):
    """integration method to be used in this calculation"""
    return scipy_integrate.quad(fun, lo, hi)[0]

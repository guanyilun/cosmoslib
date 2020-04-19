from astropy.constants import sigma_T, m_p
from astropy.cosmology import z_at_value, Planck18_arXiv_v2 as cosmo
from astropy import units as u

# CMB temperature
Tcmb  = 2.726e6  #\muk

# speed of light
c = 299792.  #km/s
# c = 2.99792458e8  # m/s
c2 = c**2  #(km/s)^2

# hubble constant from planck 2018
h = cosmo.H(0).value/100.

# critical density
crit_dens_0 = cosmo.critical_density0.to(u.solMass/u.Mpc**3).value  #M_sun/Mpc**3

# planck constant
h = 6.62607004e-34 #m^2kg/s

# boltzman constant
kB = 1.38064852e-23 #m^2kg/s^2/K


# Below are in planck units
# taken from https://github.com/marius311/Cosmology.jl/blob/master/src/PhysicalConstants.jl
# Thomson scattering cross section
sigmaT = 2.546617863799246e41

# Mass of the Hydrogen atom
mH = 7.690195407123562e-20

# Mass of the Helium atom
mHe = 3.0538317277758185e-19

# Mass of the proton
mp = 7.685126339343148e-20

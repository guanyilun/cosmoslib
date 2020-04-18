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

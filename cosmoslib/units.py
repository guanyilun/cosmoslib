"""Units commonly used"""


##########################
# cosmology: planck unit #
##########################

class planck:
    km = 6.187152231383511e37
    second = 1.8548615754666474e43
    kelvin = 7.058235349009319e-33
    eV = 8.190723190477751e-29
    Mpc = 1.90916956403801e57
    gram = 45946.59168182904
    joule = 5.11221307704105e-10
    aRad = 4.0*5.670400e-8 # radiation constant
    # derived
    KeV = 1e-6*eV
    MeV = 1e-3*eV
    # unit of hubble constant
    H0_unit = km/second/Mpc
    m_e = 510.9989461 * KeV

###########################
# cosmology: natural unit #
###########################

class natural:
    km = (1.9733e-16)**-1 / 1000  # GeV^-1
    cm = km * 1e-5  # GeV^-1
    second = (6.5823e-25)**-1  # GeV^-1
    kevin = 1.38e-23 / (1.602e-10)  # GeV
    eV = 1e-9
    Mpc = 3.0856e24*cm
    kg = (1.7827e-27)**-1
    gram = kg * 1000
    joule = (1.6022e-10)**-1
    KeV = 1e-6
    MeV = 1e-3
    m_e = 0.511 * MeV

    # convention that e^2 = alpha
    e = 0.0854
    Gauss = 6.925e-20  # GeV^2

    # convention that e^2 = 4\pi alpha
    # e = 0.303  # 0.0854 * sqrt(4\pi)
    # Gauss = 1.95e-20  # GeV^2

    # unit of hubble constant
    H0_unit = km/second/Mpc
    nG = 1e-9*Gauss
    Hz = 1/second
    GHz = 1e9*Hz


#######
# map #
#######

arcmin = 0.000290888

"""Notes + codes related to Navarro-Frenk-White (NFW) cluster mass profile

Arxiv 1412.7521: Eq. 5: mass density profile

\rho(r) = [ 200/3*c^3 / (ln(1+c)-c/(1+c)) ] [ \rho_crit / (rc/r_200)(1+rc/r_200) ]
        = [ 4\rho_s r_s^3 / r(r+r_s)^2 ]

where r_s = r_200/c_200, \rho_s = \rho(r_s)

Lensing deflection angle for NFW profile (Eq. 6):

\vec{\delta\theta} = 16\pi GA/(c*r_200) [\vec{\theta} / \theta] * [d_SL / d_S]
                   * f(d_L\theta c/r_200)

where

f(x) = 1/x { ln(x/2) + [ ln(x/(1-sqrt(1-x^2))) / sqrt{1-x^2} ] }  for x<1
     = 1/x { ln(x/2) + [ (\pi/2-arcsin(1/x)) / sqrt{x^2-1} ] }    for x>1

and

A = [ M_200 c^2 / (4\pi * (ln(1+c)-c/(1+c))) ]

This gives

\vec{\delta\theta} = 16\pi G [ M_200 c^2 / (4\pi * (ln(1+c)-c/(1+c))) ] / (c_200*r_200)
                   * [d_SL / d_S] f(d_L\theta c/r_200) \hat{\theta}

In the limit when the lense is very close to us, d_SL / d_S ~ 1. Therefore,

\vec{r} = 16\pi G [ M_200 / (4\pi * (ln(1+c)-c/(1+c))) ] / r_s * f(r/r_s) \hat{r}

"""

def rho_3D(r, r_s, rho_s):
    return 4*rho_s*r_s**3/(r*(r+r_s)**2)

def delta_theta(r, r_s, r_200, c_200, M_200):
    A = M_200 / (4*np.pi*(np.log(1+c_200)-c_200/(1+c_200)))
    def f(x):
        fx = 1/x
        if x > 1:
            fx *= np.log(x/2) + np.log(x/(1-np.sqrt(1-x**2)))/np.sqrt(1-x**2)
        else:
            fx *= np.log(x/2) + (np.pi/2-np.arcsin(1/x))/np.sqrt(x**2-1)
        return fx
    # deflection
    rs = r_200 / c_200
    delta = 16*np.pi*G/r_s * f(r/r_s)
    return delta

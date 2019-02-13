"""Reusable functions related to camb and power spectrum calculation

This module collects some reusable functions that I used when working
with camb, power spectrum and covariance matrix
"""

from matplotlib import pyplot as plt
import numpy as np
from camb import model, initialpower
import camb
import cambex


def generate_camb_params(ombh2, omch2, tau, ln10e10As, ns, omk=0,
                         lmax=5000, H0=67.11, r=0):
    """A wrapper for camb parameter generator.
    
    Args:
        cosmological parameters

    Returns;
        camb.CAMBparams
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=omk,
                       tau=tau)
    pars.InitPower.set_params(As=np.exp(ln10e10As)*10**-10, ns=ns,
                              r=r)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    return pars


def generate_cmb_power_spectra(pars, raw_cl=True):
    """A wrapper for CAMB function. 

    Default unit of the spectrais muK, and by default the total power
    spectra is used.  

    Args:

        pars: camb.CAMBparams
        raw_cl: whether l(l+1)/2\pi prefactor is included, True means
                will not include the prefactor

    Returns:
        ClTT, ClEE, ClBB, ClTE

    """
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK',
                                           raw_cl=raw_cl)
    tot = powers['total']

    # get the number of ells computed
    n_ell = tot.shape[0]

    # separate different spectra better readability
    ClTT = tot[:,0]
    ClEE = tot[:,1]
    ClBB = tot[:,2]
    ClTE = tot[:,3]
    
    return ClTT, ClEE, ClBB, ClTE


def TGC_to_TEB(ClTT, ClGG, ClCC, ClTG):
    """Transform TGC convention to TEB convension"""
    return ClTT, 2*ClGG, 2*ClCC, np.sqrt(2)*ClTG


def TEB_to_TGC(ClTT, ClEE, ClBB, ClTE):
    """Transform TEB convention to TGC convension"""
    return ClTT, 0.5*ClEE, 0.5*ClBB, 1.0/np.sqrt(2)*ClTE

    
def generate_power_spectra_realization(pars, raw_cl=True):
    """Generate a realization of the power spectra

    Args:
        pars: CAMB param
        raw_cl: True if l(l+1)/2\pi prefactor is not included

    Returns:
        ClTT, ClEE, ClBB, ClTE power spectra
    """
    ClTT, ClEE, ClBB, ClTE = generate_cmb_power_spectra(pars, raw_cl=raw_cl)

    # convert into TGC convention for easier calculation
    ClTT, ClGG, ClCC, ClTG = TEB_to_TGC(ClTT, ClEE, ClBB, ClTE)

    # define empty arrays to hold the generated power spectra
    lmax = len(ClTT)
    m_ClTT = np.zeros(lmax)
    m_ClGG = np.zeros(lmax)
    m_ClCC = np.zeros(lmax)
    m_ClTG = np.zeros(lmax)

    # this is certainly slow, but I can use it as a benchmark
    # to see how much acceleration I can get by changing into
    # a compiled version. A quick test shows that it takes >5
    # minutes to run! Note that we start from l=2 because l=0,1
    # are 0 for convenience
    for l in range(2, lmax):
        # a faster version
        zeta1 = np.random.normal(0, 1, 2*l+1)*np.exp(1j*np.random.uniform(0, 2*np.pi, 2*l+1))
        zeta2 = np.random.normal(0, 1, 2*l+1)*np.exp(1j*np.random.uniform(0, 2*np.pi, 2*l+1))
        zeta3 = np.random.normal(0, 1, 2*l+1)*np.exp(1j*np.random.uniform(0, 2*np.pi, 2*l+1))

        # generate alm
        aTlm = zeta1 * ClTT[l]**0.5
        aGlm = zeta1 * ClTG[l] / (ClTT[l])**0.5 + zeta2*(ClGG[l] - ClTG[l]**2/ClTT[l])
        aClm = zeta3 * ClCC[l]**0.5

        i_ClTT = np.sum(1.0 * np.abs(aTlm)**2 / (2*l+1))
        i_ClGG = np.sum(1.0 * np.abs(aGlm)**2 / (2*l+1))
        i_ClCC = np.sum(1.0 * np.abs(aClm)**2 / (2*l+1))
        i_ClTG = np.sum(1.0 * np.conj(aTlm)*aGlm / (2*l+1))

        m_ClTT[l] = i_ClTT
        m_ClGG[l] = i_ClGG
        m_ClCC[l] = i_ClCC
        m_ClTG[l] = i_ClTG

    return TGC_to_TEB(m_ClTT, m_ClGG, m_ClCC, m_ClTG)


def add_noise(ClTT, ClEE, ClBB, ClTE, pixel_noise, theta_fwhm):
    """ with noises added in to
    simulate real data
    
    Args:
        ClTT, ClEE, ClBB, ClTE: power spectra
        pixel_noise: noise per pixel
        theta_fwhm: beam size in degress (full width half minimum)
        f_sky: sky coverage fraction, 1 means full-sky coverage

    Returns:
        ClTT, ClEE, ClBB, ClTE power spectra
    """
    # first convert to the notations consistent with Kamionkowski,
    # Kosowsky, Stebbins (1997)
    ClTT, ClGG, ClCC, ClTG = TEB_to_TGC(ClTT, ClEE, ClBB, ClTE)

    sigma_b = 0.00742 * theta_fwhm 

    # assuming the beam is a gaussian beam with an ell dependent
    # beam size
    Wb = lambda l: np.exp(-l*(l+1)*sigma_b**2/2)

    # calculate the noise parameter w^-1
    winv = pixel_noise**2

    ell = np.arange(len(ClTT))
    
    ClTT += winv * np.abs(Wb(ell))**-2
    ClGG += winv * np.abs(Wb(ell))**-2
    ClCC += winv * np.abs(Wb(ell))**-2
    # ClTG remain unchanged

    # convert back
    return TEB_to_TGC(ClTT, ClGG, ClCC, ClTG)


def covariance_matrix(pars, pixel_noise, theta_fwhm, f_sky=1):
    """Calculate the covariance matrix based on a model.
    22
    Args:
        pars: cambParams object
        pixel_noise: noise per pixel
        theta_fwhm: beam size in degress (full width half minimum)
        f_sky: sky coverage fraction, 1 means full-sky coverage

    Returns:
        cov: a tensor of size [n_ell, n_ps, n_ps], for example with
             a lmax of 5000, the tensor size will be [5000, 4, 4]
    """
    sigma_b = 0.00742 * theta_fwhm 

    # assuming the beam is a gaussian beam with an ell dependent
    # beam size
    Wb = lambda l: np.exp(-l*(l+1)*sigma_b**2/2)

    # calculate the noise parameter w^-1
    winv = pixel_noise**2

    # calculate the cmb power spectra based on the parameter
    ClTT, ClEE, ClBB, ClTE = generate_cmb_power_spectra(pars, raw_cl=True)

    # initialize empty covariance tensor. Since the covariance
    # matrix depends on ell, we will make a higher dimensional array
    # [n_ell, n_ps, n_ps] where the first index represents
    # different ells, the second and third parameters represents
    # different power spectra
    n_ell = len(ClTT)    
    cov = np.zeros([n_ell, 4, 4])

    # TODO: this can be done more efficiently by vectorization
    for l in range(n_ell):
        # T, T 
        cov[l,0,0] = 2.0/(2*l+1)*(ClTT[l] + winv*np.abs(Wb(l))**-2)**2

        # E, E 
        cov[l,1,1] = 2.0/(2*l+1)*(0.5*ClEE[l] + winv*np.abs(Wb(l))**-2)**2 

        # B, B 
        cov[l,2,2] = 2.0/(2*l+1)*(0.5*ClBB[l] + winv*np.abs(Wb(l))**-2)**2 

        # TE, TE 
        cov[l,3,3] = 1.0/(2*l+1)*(0.5*ClTE[l]**2 + (ClTT[l] + winv*np.abs(Wb(l))**-2)
                                   *(0.5*ClEE[l] + winv*np.abs(Wb(l))**-2))

        # T, E 
        cov[l,0,1] = cov[l,1,0] = 2.0/(2*l+1)*0.5*ClTE[l]**2

        # T, B 
        cov[l,0,2] = cov[l,2,0] = 0

        # T, TE
        cov[l,0,3] = cov[l,3,0] = 2.0/(2*l+1)*(0.5*ClTE[l]*(ClTT[l] + winv*np.abs(Wb(l))**-2))

        # E, TE 
        cov[l,1,3] = cov[l,3,1] = 2.0/(2*l+1)*(1.0/np.sqrt(2)*ClTE[l]*(0.5*ClEE[l] + winv*np.abs(Wb(l))**-2))
    
        # now we include the effect of partial sky coverage
        cov *= f_sky

    return cov


def fisher_matrix(model, cov, ratio=0.01):
    """Estimate the fisher matrix near the model provided. 

    Args: 
        model: a python dictionary that contains the cosmological
        parameters that one is interested in estimating the fisher
        matrix between them

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


def generate_ps(ombh2=0.02225, omch2=0.1198, hubble=67.8):
    """Generate the total power spectrum from primary and magnetic
    field contribution"""

    BASE_DIR = "/home/aaron/Workspace/research/MagCAMB/"

    # open a magcamb session
    magcamb = cambex.CambSession(camb_bin=BASE_DIR+'camb')

    # define base parameters
    magcamb.set_base_params(BASE_DIR+"custom/params/params_mag.ini")

    # set cosmological parameters
    magcamb.ini.set("ombh2", ombh2)
    magcamb.ini.set("omch2", omch2)
    magcamb.ini.set("hubble", hubble)
    # more can be set here

    ###################################
    # passive scalar and tensor modes #
    ###################################

    # these two modes can be computed together
    magcamb.ini.set("get_scalar_cls", "T")
    magcamb.ini.set("get_vector_cls", "F")
    magcamb.ini.set("get_tensor_cls", "T")
    magcamb.ini.set("magnetic_mode", "2")

    magcamb.run()

    # collect power spectra
    passive_scalar = magcamb.load_scalarCls().values.T
    passive_tensor = magcamb.load_tensorCls().values.T


    ###########################
    # compensated scalar mode #
    ###########################

    magcamb.ini.set("get_scalar_cls", "T")
    magcamb.ini.set("get_vector_cls", "F")
    magcamb.ini.set("get_tensor_cls", "F")
    magcamb.ini.set("magnetic_mode", "1")

    magcamb.run()

    comp_scalar = magcamb.load_scalarCls().values.T

    ###########################
    # compensated vector mode #
    ###########################

    magcamb.ini.set("get_scalar_cls", "F")
    magcamb.ini.set("get_vector_cls", "T")
    magcamb.ini.set("get_tensor_cls", "F")
    magcamb.ini.set("magnetic_mode", "1")

    magcamb.run()

    comp_vector = magcamb.load_vectorCls().values.T

    ##################################
    # get primary cmb power spectrum #
    ##################################

    # open a camb session
    CAMB_DIR = '/home/aaron/Workspace/research/CAMB-0.1.7/'
    camb = cambex.CambSession(camb_bin=CAMB_DIR+'camb')

    # define base parameter
    PARAM_DIR = '/home/aaron/Workspace/research/MagCAMB/'
    camb.set_base_params(PARAM_DIR+"custom/params/params.ini")

    # set cosmological parameters
    camb.ini.set("ombh2", "0.02225")
    camb.ini.set("omch2", "0.1198")
    camb.ini.set("hubble", "67.8")

    # these two modes can be computed together
    camb.ini.set("get_scalar_cls", "T")
    camb.ini.set("get_vector_cls", "F")
    camb.ini.set("get_tensor_cls", "T")

    # run the session
    camb.run()

    # get the scalar mode
    primary_scalar = camb.load_scalarCls().values.T
    primary_tensor = camb.load_tensorCls().values.T
    primary_lensed = camb.load_lensedCls().values.T

    # total power spectrum
    # generate an empty array to hold the total power spectra
    ps = [primary_lensed, passive_scalar, passive_tensor,
          comp_scalar, comp_vector]
    N = min([p.shape[0] for p in ps])

    total_ps = np.zeros((N,5))

    # ell
    total_ps[:, 0] = primary_lensed[:N, 0]

    # ClTT
    total_ps[:, 1] = primary_lensed[:N, 1] + passive_scalar[:N, 1] + \
                     passive_tensor[:N, 1] + comp_scalar[:N, 1] + \
                     comp_vector[:N, 1] 

    # ClEE
    total_ps[:, 2] = primary_lensed[:N, 2] + passive_scalar[:N, 2] + \
                     passive_tensor[:N, 2] + comp_scalar[:N, 2] + \
                     comp_vector[:N, 2]

    # ClBB
    total_ps[:, 3] = primary_lensed[:N, 3] + passive_tensor[:N, 3] + \
                     comp_vector[:N, 3]

    # ClTE
    total_ps[:, 4] = primary_lensed[:N, 4] + passive_scalar[:N, 3] + \
                     passive_tensor[:N, 4] + comp_scalar[:N, 3] + \
                     comp_vector[:N, 4]

    return total_ps

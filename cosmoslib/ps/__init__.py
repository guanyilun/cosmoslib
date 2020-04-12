from .ps import PS, SimpleNoise, add_prefactor, remove_prefactor, \
    resample, N_l, add_noise_nl, Dl2Cl, Cl2Dl, gen_ps_realization, \
    covmat_slow, fisher_matrix

try: from ._ps import covmat
except: covmat = covmat_slow

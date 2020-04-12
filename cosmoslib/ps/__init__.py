from .ps import PS, SimpleNoise, add_prefactor, remove_prefactor, \
    resample, N_l, add_noise_nl, Dl2Cl, Cl2Dl, gen_ps_realization, \
    covmat as covmat_slow, fisher_matrix

try:
    from ._ps import covmat as covmat_fast
    covmat = covmat_fast
except: covmat = covmat_slow

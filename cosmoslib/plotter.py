from matplotlib import pyplot as plt
import numpy as np
from cycler import cycler

def plot_ps(total_ps, fmt="-", axes=None, **kwargs):
    """Plot the power spectra"""
    if not np.any(axes):
        fig, axes = plt.subplots(2,2,figsize=(12,9))

    if total_ps.shape[-1] == 5:
        axes[0,0].loglog(total_ps[:,0],total_ps[:,1], fmt, **kwargs)
        axes[0,1].loglog(total_ps[:,0],total_ps[:,2], fmt, **kwargs)
        axes[1,0].loglog(total_ps[:,0],total_ps[:,3], fmt, **kwargs)
        axes[1,1].loglog(total_ps[:,0],np.abs(total_ps[:,4]), fmt, **kwargs)
    elif total_ps.shape[-1] == 4:
        axes[0,0].loglog(total_ps[:,0],total_ps[:,1], fmt, **kwargs)
        axes[0,1].loglog(total_ps[:,0],total_ps[:,2], fmt, **kwargs)
        axes[1,1].loglog(total_ps[:,0],np.abs(total_ps[:,3]), fmt, **kwargs)

    for i, label in enumerate(['TT','EE','BB','TE']):
        ax = axes[i//2,i%2]
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(r'$\ell(\ell+1)C_\ell^{\rm %s}/2\pi$' % label)
        ax.legend()

    return axes

def set_plotstyle(options={}, style='default', tex=None):
    """Define common plot style"""
    import seaborn as sns
    import matplotlib.pyplot as plt

    default = {}
    if style == 'default':
        # style from Cristobal
        for tick in ('xtick', 'ytick'):
            default['{0}.major.size'.format(tick)] = 8
            default['{0}.minor.size'.format(tick)] = 4
            default['{0}.major.width'.format(tick)] = 1
            default['{0}.minor.width'.format(tick)] = 1
            default['{0}.labelsize'.format(tick)] = 14
            default['{0}.direction'.format(tick)] = 'in'
        default['xtick.top'] = True
        default['ytick.right'] = True
        default['axes.linewidth'] = 1
        default['axes.labelsize'] = 14
        default['font.size'] = 14
        default['font.family']='sans-serif'
        default['legend.fontsize'] = 14
        default['lines.linewidth'] = 2
        default['axes.prop_cycle'] = cycler(color=['#2424f0','#df6f0e','#3cc03c','#d62728','#b467bd','#ac866b','#e397d9','#9f9f9f','#ecdd72','#77becf'])
    elif style == 'ap':
        # from astropaint
        plt.style.use("seaborn-poster")
        default["figure.figsize"] = (6, 4)
        default["text.usetex"] = True
        default["font.size"] = 16
        default["font.family"] = "serif"
        default['font.serif'] = 'Ubuntu'
        default["figure.dpi"]= 100
    elif style == 'clp':
        # from cmblensplus
        default['axes.labelsize'] = 8
        default['legend.fontsize'] = 10
        default['xtick.labelsize'] = 10
        default['ytick.labelsize'] = 10
        default['text.usetex'] = False
    else:  # try to load matplotlib internal styles
        plt.style.use(style)
    for key in default:
        plt.rcParams[key] = default[key]
    # overwrite if necessary
    for key in options:
        plt.rcParams[key] = options[key]
    if tex is not None:
        plt.rcParams['text.usetex'] = tex
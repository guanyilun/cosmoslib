from matplotlib import pyplot as plt

def plot_ps(total_ps, axes=None, **kwargs):
    """Plot the power spectra"""
    if not np.any(axes):
        fig, axes = plt.subplots(2,2,figsize=(12,9))

    axes[0,0].loglog(total_ps[:,0],total_ps[:,1], **kwargs)
    axes[0,1].loglog(total_ps[:,0],total_ps[:,2], **kwargs)
    axes[1,0].loglog(total_ps[:,0],total_ps[:,3], **kwargs)
    axes[1,1].loglog(total_ps[:,0],np.abs(total_ps[:,4]), **kwargs)

    for i, label in enumerate(['TT','EE','BB','TE']):
        ax = axes[i//2,i%2]
        ax.set_xlabel('$\ell$')
        ax.set_ylabel('$\ell(\ell+1)C_l^{%s}/2\pi$' % label)
        ax.legend()

    return axes


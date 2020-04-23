from matplotlib import pyplot as plt
import numpy as np

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

def set_plotstyle(style='default', options={}):
    """Define common plot style"""
    import seaborn as sns
    import matplotlib.pyplot as plt

    default = {}
    if style == 'default':
        # style from Cristobal
        for tick in ('xtick', 'ytick'):
            default['{0}.major.size'.format(tick)] = 8
            default['{0}.minor.size'.format(tick)] = 4
            default['{0}.major.width'.format(tick)] = 2
            default['{0}.minor.width'.format(tick)] = 2
            default['{0}.labelsize'.format(tick)] = 20
            default['{0}.direction'.format(tick)] = 'in'
        default['xtick.top'] = True
        default['ytick.right'] = True
        default['axes.linewidth'] = 2
        default['axes.labelsize'] = 22
        default['font.size'] = 22
        default['font.family']='sans-serif'
        default['legend.fontsize'] = 18
        default['lines.linewidth'] = 2
        default['axes.prop_cycle'] = "cycler(color=['#2424f0','#df6f0e','#3cc03c','#d62728','#b467bd','#ac866b','#e397d9','#9f9f9f','#ecdd72','#77becf'])"
    elif style == 's1':
        "style stolen from astropaint"

        plt.style.use("seaborn-poster")
        default["figure.figsize"] = (6, 4)
        default["text.usetex"] = True
        default["font.size"] = 16
        default["font.family"] = "serif"
        default['font.serif'] = 'Ubuntu'
        default["figure.dpi"]= 100

    for key in default:
        plt.rcParams[key] = default[key]
    # overwrite if necessary
    for key in options:
        plt.rcParams[key] = options[key]


class Plotter(object):
    """Nice plotter adapted from orphics"""
    def __init__(self,scheme=None,xlabel=None,ylabel=None,xyscale=None,xscale="linear",
                 yscale="linear",ftsize=14,thk=1,labsize=None,major_tick_size=5,
                 minor_tick_size=3,scalefn=None,**kwargs):
        self.scalefn = None
        if scheme is not None:
            if scheme=='Dell' or scheme=='Dl':
                xlabel = '$\\ell$' if xlabel is None else xlabel
                ylabel = '$D_{\\ell}$' if ylabel is None else ylabel
                xyscale = 'linlog' if xyscale is None else xyscale
                self.scalefn = (lambda x: x**2./2./np.pi) if scalefn is None else scalefn
            elif scheme=='Cell' or scheme=='Cl':
                xlabel = '$\\ell$' if xlabel is None else xlabel
                ylabel = '$C_{\\ell}$' if ylabel is None else ylabel
                xyscale = 'linlog' if xyscale is None else xyscale
                self.scalefn = (lambda x: 1)  if scalefn is None else scalefn
            elif scheme=='CL':
                xlabel = '$L$' if xlabel is None else xlabel
                ylabel = '$C_{L}$' if ylabel is None else ylabel
                xyscale = 'linlog' if xyscale is None else xyscale
                self.scalefn = (lambda x: 1)  if scalefn is None else scalefn
            elif scheme=='LCL':
                xlabel = '$L$' if xlabel is None else xlabel
                ylabel = '$LC_{L}$' if ylabel is None else ylabel
                xyscale = 'linlin' if xyscale is None else xyscale
                self.scalefn = (lambda x: x)  if scalefn is None else scalefn
            elif scheme=='rCell' or scheme=='rCl':
                xlabel = '$\\ell$' if xlabel is None else xlabel
                ylabel = '$\\Delta C_{\\ell} / C_{\\ell}$' if ylabel is None else ylabel
                xyscale = 'linlin' if xyscale is None else xyscale
                self.scalefn = (lambda x: 1)  if scalefn is None else scalefn
            elif scheme=='dCell' or scheme=='dCl':
                xlabel = '$\\ell$' if xlabel is None else xlabel
                ylabel = '$\\Delta C_{\\ell}$' if ylabel is None else ylabel
                xyscale = 'linlin' if xyscale is None else xyscale
                self.scalefn = (lambda x: 1)  if scalefn is None else scalefn
            elif scheme=='rCL':
                xlabel = '$L$' if xlabel is None else xlabel
                ylabel = '$\\Delta C_{L} / C_{L}$' if ylabel is None else ylabel
                xyscale = 'linlin' if xyscale is None else xyscale
                self.scalefn = (lambda x: 1)  if scalefn is None else scalefn
            else:
                raise ValueError
        if self.scalefn is None:
            self.scalefn = (lambda x: 1) if scalefn is None else scalefn
        if xyscale is not None:
            scalemap = {'log':'log','lin':'linear'}
            xscale = scalemap[xyscale[:3]]
            yscale = scalemap[xyscale[3:]]
        matplotlib.rc('axes', linewidth=thk)
        matplotlib.rc('axes', labelcolor='k')
        self.thk = thk
        self._fig=plt.figure(**kwargs)
        self._ax=self._fig.add_subplot(1,1,1)
        # Some self-disciplining :)
        try:
            force_label = os.environ['FORCE_ORPHICS_LABEL']
            force_label = True if force_label.lower().strip() == "true" else False
        except:
            force_label = False
        if force_label:
            assert xlabel is not None, "Please provide an xlabel for your plot"
            assert ylabel is not None, "Please provide a ylabel for your plot"
        if xlabel!=None: self._ax.set_xlabel(xlabel,fontsize=ftsize)
        if ylabel!=None: self._ax.set_ylabel(ylabel,fontsize=ftsize)
        self._ax.set_xscale(xscale, nonposx='clip')
        self._ax.set_yscale(yscale, nonposy='clip')

        if labsize is None: labsize=ftsize-2
        plt.tick_params(axis='both', which='major', labelsize=labsize,width=self.thk,size=major_tick_size)#,size=labsize)
        plt.tick_params(axis='both', which='minor', labelsize=labsize,size=minor_tick_size)#,size=labsize)
        self.do_legend = False

    def legend(self,loc='best',labsize=12,numpoints=1,**kwargs):
        self.do_legend = False
        handles, labels = self._ax.get_legend_handles_labels()
        legend = self._ax.legend(handles, labels,loc=loc,prop={'size':labsize},numpoints=numpoints,frameon = 1,**kwargs)
        # return legend
        return self

    def add(self,x,y,label=None,lw=2,linewidth=None,addx=0,**kwargs):
        if linewidth is not(None): lw = linewidth
        if label is not None: self.do_legend = True
        scaler = self.scalefn(x)
        yc = y*scaler
        self._ax.plot(x+addx,yc,label=label,linewidth=lw,**kwargs)
        return self

    def hist(self,data,**kwargs):
        return self._ax.hist(data,**kwargs)

    def add_err(self,x,y,yerr,ls='none',band=False,alpha=1.,marker="o",elinewidth=2,markersize=4,label=None,mulx=1.,addx=0.,**kwargs):
        scaler = self.scalefn(x)
        yc = y*scaler
        yerrc = yerr*scaler
        if band:
            self._ax.plot(x*mulx+addx,yc,ls=ls,marker=marker,label=label,markersize=markersize,**kwargs)
            self._ax.fill_between(x*mulx+addx, yc-yerrc, y+yerrc, alpha=alpha)
        else:
            self._ax.errorbar(x*mulx+addx,yc,yerr=yerrc,ls=ls,marker=marker,elinewidth=elinewidth,markersize=markersize,label=label,alpha=alpha,**kwargs)
        if label is not None: self.do_legend = True
        return self

    def plot2d(self,data,lim=None,levels=None,clip=0,clbar=True,cm=None,label=None,labsize=14,extent=None,ticksize=12,**kwargs):
        '''
        For an array passed in as [j,i]
        Displays j along y and i along x , so (y,x)
        With the origin at upper left
        '''
        Nx=data.shape[0]
        Ny=data.shape[1]
        arr=data[clip:Nx-clip,clip:Ny-clip]

        if type(lim) is list or type(lim) is tuple:
            limmin,limmax = lim
        elif lim is None:
            limmin=None
            limmax = None
        else:
            limmin=-lim
            limmax = lim
        img = self._ax.imshow(arr,interpolation="none",vmin=limmin,vmax=limmax,cmap=cm,extent=extent,**kwargs)
        if levels!=None:
           self._ax.contour(arr,levels=levels,extent=extent,origin="upper",colors=['black','black'],linestyles=['--','-'])
        if clbar:
            cbar = self._fig.colorbar(img)
            for t in cbar.ax.get_yticklabels():
                t.set_fontsize(ticksize)
            if label!=None:
                cbar.set_label(label,size=labsize)#,rotation=0)
    def hline(self,y=0.,ls="--",alpha=0.5,color="k",**kwargs):
        self._ax.axhline(y=y,ls=ls,alpha=alpha,color=color,**kwargs)
        return self

    def vline(self,x=0.,ls="--",alpha=0.5,color="k",**kwargs):
        self._ax.axvline(x=x,ls=ls,alpha=alpha,color=color,**kwargs)
        return self

    def done(self,filename=None,verbose=True,**kwargs):
        if self.do_legend: self.legend()
        if filename is not None:
            self._fig.savefig(filename,bbox_inches='tight',**kwargs)
            if verbose: cprint("Saved plot to "+ filename,"g")
        else:
            plt.show()
        plt.close(self._fig)

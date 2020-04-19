"""utility functions related to binning"""
import math, numpy as np


class Binning():
    def __init__(self, lmin, lmax, nbins, scheme='linear', w_func=None):
        """create a binning object that has everything needed
        to bin an unbinned power-spectra

        Parameters:
        -----------
        lmin / lmax (int): bounds of ells to bin (inclusive)
        nbins: number of bins to create
        scheme: binning scheme, i.e. 'linear', 'log', etc.
        w_func: w_func(ell) -> weights for ell, default to 1

        """
        self.lmin, self.lmax = lmin, lmax
        self.nbins, self.schemes = nbins, schemes
        self.w_func = w_func if w_func else lambda ells: np.ones_like(ells)

        # get ells and weights
        self.ells = np.arange(lmin, lmax+1)
        try: weights = w_func(ells)  # see if vectorization work
        except: weights = np.array([w_func(ell) for ell in ells])

        self.create_bins()

    def create_bins():
        lmin, lmax, nbins = self.lmin, self.lmax, self.nbins
        scheme = self.scheme
        # find bin egdes and centers
        if scheme = 'linear':
            edges = np.linspace(lmin, lmax, nbins+1)
        elif scheme = 'log':
            edges = np.logspace(lmin, lmax, nbins+1, base=math.e)
        elif scheme = 'log10':
            edges = np.logspace(lmin, lmax, nbins+1, base=10)
        elif scheme = 'p2':  # quadratic
            edges = np.linspace(np.sqrt(lmin), np.sqrt(lmax), nbins+1)**2
        elif scheme = 'p3':  # cubic
            edges = np.linspace(lmin**(1/3), lmax**(1/3), nbins+1)**3
        else:
            raise NotImplementedError(f'Binning scheme: {scheme} not supported!')
        self.bin_l, self.bin_r = edges[:-1], edges[1:]
        self.bin_c = (self.bin_r - self.bin_l) / 2
        self.slices = [None]*nbins  # store slices to extract ells
        for l, r in zip(self.bin_l, self.bin_r):
            idx_start = np.where(self.ells>=l)[0][0]  # first of >= bin_l
            idx_end = np.where(self.ells<r)[0][-1]    # last of < bin_r
            self.slices[i] = slice(idx_start, idx_end, None)

    @classmethod
    def for_ps(cls, ps, nbins, scheme='linear', w_func=None):
        return cls(ps.lmin, ps.lmax, nbins, scheme, w_func)


class BinnedPS:
    def __init__(self, ps, binning):
        """Binned power spectrum object.

        Parameters:
        -----------
        ps: power spectrum object
        binning: binning object

        """
        # check for ell range matching
        if ps.lmin != binning.lmin or ps.lmax != binning.lmax:
            raise ValueError("ps and binning are incompatible!")

        # store a reference to unbinned_ps and binning
        self.unbinned_ps = ps
        self.binning = binning
        self.ps = {}

        # bin each spectra
        w = binning.weights
        for spec in ps.specs:
            cl = ps.ps[spec]
            bcl = np.zeros(binning.nbins)
            for i in range(binning.nbins):
                s = binning.slices[i]
                bcl[i] = np.sum(cl[s]*w[s])/np.sum(w[s])
            self.ps[spec] = bcl
        self.ells = binning.bin_c

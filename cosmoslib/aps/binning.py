"""utility functions related to binning"""
import math, numpy as np
from .ps import PS, Covmat


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
        self.nbins, self.scheme = nbins, scheme
        self.w_func = w_func if w_func else lambda ells: np.ones_like(ells)

        # get ells and weights
        self.ells = np.arange(lmin, lmax+1)
        try: self.weights = self.w_func(self.ells)  # see if vectorization work
        except: self.weights = np.array([self.w_func(ell) for ell in self.ells])

        # find bin egdes and centers
        if scheme == 'linear':
            edges = np.linspace(lmin, lmax, nbins+1)
        elif scheme == 'log':
            edges = np.exp(np.linspace(np.log(lmin), np.log(lmax), nbins+1))
        elif scheme == 'log10':
            edges = 10**(np.linspace(np.log10(lmin), np.log10(lmax), nbins+1))
        elif scheme == 'p2':  # quadratic
            edges = np.linspace(np.sqrt(lmin), np.sqrt(lmax), nbins+1)**2
        elif scheme == 'p3':  # cubic
            edges = np.linspace(lmin**(1/3), lmax**(1/3), nbins+1)**3
        else:
            raise NotImplementedError(f'Binning scheme: {scheme} not supported!')
        self.bin_l, self.bin_r = edges[:-1], edges[1:]
        self.bin_c = (self.bin_r + self.bin_l) / 2
        self.slices = [None]*nbins  # store slices to extract ells
        for i, (l, r) in enumerate(zip(self.bin_l, self.bin_r)):
            idx_start = np.where(self.ells>=l)[0][0]  # first of >= bin_l
            idx_end = np.where(self.ells<r)[0][-1]    # last of < bin_r
            self.slices[i] = slice(idx_start, idx_end+1, None)

    @classmethod
    def for_ps(cls, ps, nbins, scheme='linear', w_func=None):
        return cls(ps.lmin, ps.lmax, nbins, scheme, w_func)

    def __repr__(self):
        return f"Binning(nbins={self.nbins},scheme={self.scheme},"\
            f"lmin={self.lmin},lmax={self.lmax})"


class BinnedPS(PS):
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
        self.order = ps.order
        # make sure we bin without prefactor of (l+1)l/2pi
        if ps.prefactor: ps.remove_prefactor()
        self.prefactor = False

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
        self.ps['ell'] = binning.bin_c  # use bin center as proxy

    def covmat(self, noise, f_sky=1):
        """Calculate covariance matrix based on a noise model that
        is based on the *unbinned* cl."""
        # we will compute the unbinned covariance matrix from stored
        # copy of ps
        assert np.allclose(noise.ell, self.unbinned_ps.ell), "ell has to match for this to work!"
        covmat = self.unbinned_ps.covmat(noise, f_sky)
        ell, cov = covmat.ell, covmat.cov
        # bin the covmat using the same binning scheme
        bin_ell = self.ell
        bin_cov = np.zeros((self.binning.nbins,)+cov.shape[1:])
        w = self.binning.weights
        # Note: covariance matrix has shape of [n_ell, n_spec, n_spec]
        for i, s in enumerate(self.binning.slices):
            # the variance goes like weighted average / n_ell in the bin
            bin_cov[i,:,:] = np.einsum('ijk,i->jk',
                                       cov[s,:,:], w[s]) / np.sum(w[s]) / len(w[s])
        return Covmat(bin_ell, bin_cov, order=covmat.order)

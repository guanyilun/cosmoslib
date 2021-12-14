import random, string
import numpy as np

def rand_chars(n):
    """Generate n random characters"""
    return ''.join(random.choice(string.ascii_letters) for x in range(n))


class Binner:
    def __init__(self, xmin, xmax, nbins, scheme='linear', w_func=None):
        self.xmin, self.xmax = xmin, xmax
        self.nbins = nbins
        self.w_func = w_func if w_func else lambda bc: np.ones_like(bc)
        self.slices = None

        # find bin egdes and centers
        if scheme == 'linear':
            edges = np.linspace(xmin, xmax, nbins+1)
        elif scheme == 'log':
            edges = np.exp(np.linspace(np.log(xmin), np.log(xmax), nbins+1))
        elif scheme == 'log10':
            edges = 10**(np.linspace(np.log10(xmin), np.log10(xmax), nbins+1))
        elif scheme == 'p2':  # quadratic
            edges = np.linspace(np.sqrt(xmin), np.sqrt(xmax), nbins+1)**2
        elif scheme == 'p3':  # cubic
            edges = np.linspace(xmin**(1/3), xmax**(1/3), nbins+1)**3
        else:
            raise NotImplementedError(f'Binning scheme: {scheme} not supported!')

        self.bin_l, self.bin_r = edges[:-1], edges[1:]
        self.bin_c = (self.bin_r + self.bin_l) / 2

    def get_slices(self, x):
        slices = [None]*self.nbins
        for i, (l, r) in enumerate(zip(self.bin_l, self.bin_r)):
            idx_start = np.where(x>=l)[0][0]  # first of >= bin_l
            idx_end = np.where(x<r)[0][-1]    # last of < bin_r
            slices[i] = slice(idx_start, idx_end+1, None)
        return slices

    def get_weights(self, xs):
        # get weights
        try: weights = self.w_func(xs)  # see if vectorization work
        except: weights = np.array([self.w_func(x) for x in xs])
        return weights

    def bin(self, x, y, reducer=np.sum, use_weight=True):
        slices = self.get_slices(x)
        weights = self.get_weights(x)
        y_bin = np.zeros_like(self.bin_c)
        for i, sl in enumerate(slices):
            if use_weight: y_bin[i] = reducer(y[sl]*weights)/reducer(weights)
            else: y_bin[i] = reducer(y[sl])
        return self.bin_c, y_bin

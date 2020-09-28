"""Interpolation tools"""
from scipy.interpolate import InterpolatedUnivariateSpline as IUS


def interpolate(limit=[], nsamps=20, method="linear", interpolator=IUS
                transformers={}, interp_kwargs={}, fun_kwargs={}):
    """interpolation decorator to replace any function with an interpolated
    version. By default it interpolate the first argument, though that
    can be easily extended when there is a need. In any case one can
    always define a lambda function with parameters reorderred.

    Parameters:
    -----------
    limit: interpolation bounds
    nsamps: number of points to sample
    method: sampling method, choose from 'linear', 'log', 'log10'
    interpolator: interpolator to use, default to InterpolatedUnivariateSpline
    transformers: dict of functions to transform forward and backward
      to a different representation of data before running interpolation,
      for example, sometimes we may want to interpolate in the log values instead
      use:
          transformers={
              'forward' :  np.log,
              'backward':  np.exp
          }
      This will transform x,y into log(x),log(y) before interpolating and return
      the results through the inverse transformation `backward`.
    interp_kwargs: kwargs to pass to interpolator
    fun_kwargs: kwargs to pass to function to interpolate

    Example:
    --------
    @interpolate(limit=[0, 10000], nsamps=100, method='linear',
    transformers={'forward', np.log, 'backward': np.exp},
    fun_kwargs={'omega_b':0.05}, interp_kwargs={'k':3})
    def get_xe(z, omega_b):
       # some work
       return xe

    get_xe(100)  # this will be using a precalculated log-interp table

    """
    assert len(limit) == 2
    if method == 'linear':
        xs = np.linspace(limit[0], limit[1], nsamps)
    elif method == 'log':
        xs = np.exp(np.linspace(np.log(limit[0]), np.log(limit[1]), nsamps))
    elif method == 'log10':
        xs = 10**(np.linspace(np.log10(limit[0]), np.log10(limit[1]), nsamps))
    else: raise NotImplementedError
    if not ('forward' in transformers and 'backward' in transformers):
        print("Warning: transformers missing, use identity...")
        identity = lambda x: x  # do nothing
        forward = identity
        backward = identity
    else:
        backward = transformers['backward']
        forward = transformers['forward']
    # actual decorator
    def _inner(fun):
        # try to vectorize first
        try: ys = fun(xs)
        except: ys = np.asarray([fun(x, **fun_kwargs) for x in xs])
        # create interpolator
        xs_t, ys_t = forward(xs), forward(ys)
        intp = interpolator(xs_t, ys_t, **interp_kwargs)
        def _interpolated(x):
            x_t = forward(x)
            return backward(intp(x))

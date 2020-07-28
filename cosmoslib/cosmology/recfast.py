from ctypes import cdll, byref, c_double, c_int64
from cosmoslib.utils import context

def get_xe(Omega_b, Omega_c, Omega_lam, H0, Tnow, Yp, Hswitch=1, Heswitch=6, Nz=1000, zstart=10000, zend=0):
    if not context.librecfast:
        raise RuntimeError("librecfast not found and is required!")
    lib = cdll.loadLibrary(context.librecfast)
    # Output
    z = np.linspace(zstart,zend,Nz+1)[1:]
    xe = (c_double*Nz)()
    # type conversion to ctypes
    Omega_b = c_double(Omega_b)
    Omega_c = c_double(Omega_c)
    Omega_lam = c_double(Omega_lam)
    H0 = c_double(H0)
    Tnow = c_double(Tnow)
    Yp = c_double(Yp)
    Hswitch = c_int64(Hswitch)
    Heswitch = c_int64(Heswitch)
    zstart = c_double(zstart)
    zend = c_double(zend)
    Nz = c_int64(Nz)
    # call recfast
    lib.get_xe_(byref(Omega_b), byref(Omega_c), byref(Omega_lam), byref(H0), byref(Tnow), byref(Yp),
                byref(Hswitch), byref(Heswitch), byref(Nz), byref(zstart), byref(zend), byref(xe))
    # convert to numpy arrays
    xe = np.ctypeslib.as_array(xe)
    return z, xe

"""Healpy related utility functions

Including rotation of QU maps based on healpy rotator.
This part of codes is Borrowed from:
https://github.com/SharperJBCA/CBASS_Simulator/blob/master/QU_Simulator_Functions.py
An example:
rot = hp.rotator.Rotator(coord=['G','C'])
# Rotate maps into IAU Celestial frame
maps['I'] = rot.rotate_map_pixel(maps['I'])
maps['Q'], maps['U'] = rotate_QU_frame(maps['Q'],maps['U'], rotator=rot)
maps['U'] *= -1 # CMB to IAU convention

It also includes
"""

import numpy as np
import healpy as hp

def rotateQU(Q,U,pa):
    """
    Clockwise rotation matrix.
    """
    Qp = Q*np.cos(2*pa) - U*np.sin(2*pa)
    Up = Q*np.sin(2*pa) + U*np.cos(2*pa)
    return Qp,Up

def rotate_QU_frame(q, u, rot):
    """
    Rotate coordinate from of QU angles on a HEALPIX map

    Parameters
    ----------
    q, u: healpy map for stoke parameters Q and U
    rot: Rotator object

    """
    nside = int(np.sqrt(q.size/12.))
    pix = np.arange(12*nside**2).astype(int)
    vecs = np.array(hp.pix2vec(nside, pix))

    # First rotate the map pixels
    qp = rot.rotate_map_pixel(q)
    up = rot.rotate_map_pixel(u)
    qp[qp < -1e25] = hp.UNSEEN
    up[up < -1e25] = hp.UNSEEN

    # Then caculate the reference angles in the rotated frame
    vecs_r = rot(vecs,inv=True)
    angles = rot.angle_ref(vecs_r)

    L_map = (qp + 1j * up)*np.exp(1j*2*angles)

    return np.real(L_map), np.imag(L_map)

def vec2pix(nside):
    """Return a vector to pixel converter for a given nside"""
    return lambda x, y, z: hp.pixelfunc.vec2pix(nside, x, y, z)

def coord_transform(imap, coord=['C','G'], cosmo2iau=False):
    """Transform IQU maps into new coordinates, this includes
    a simple rotation of I map and a more complex rotation of QU
    maps to refer to the new axes.

    """
    from healpy.rotator import Rotator
    assert imap.shape[0] == 3 or len(imap) == 3, "Wrong shape!"
    rotator = Rotator(coord=coord)
    I_rot = rotator.rotate_map_pixel(imap[0])
    Q_rot, U_rot = rotate_QU_frame(imap[1], imap[2], rot=rotator)
    if cosmo2iau: U_rot *= -1
    return np.vstack([I_rot, Q_rot, U_rot])

class CartesionProj(hp.projector.CartesianProj):
    """Wrapper for CartesianProj to avoid keeping
    track of vec2pix function"""
    def __init__(self, nside, *args, **kwargs):
        self.nside = nside
        super().__init__(*args, **kwargs)
    def projmap(self, imap):
        if imap.shape[0] == 3 or len(imap) == 3:
            return np.stack([self.projmap(m) for m in imap], axis=0)
        return super().projmap(imap, vec2pix(self.nside))

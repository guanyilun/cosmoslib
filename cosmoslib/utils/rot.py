"""Rotate QU maps, based on healpy rotator.

Borrowed from:

https://github.com/SharperJBCA/CBASS_Simulator/blob/master/QU_Simulator_Functions.py

An example:

rot = hp.rotator.Rotator(coord=['G','C'])

# Rotate maps into IAU Celestial frame
maps['I'] = rot.rotate_map_pixel(maps['I'])
maps['Q'],maps['U'] = rotate_QU_frame(maps['Q'],maps['U'], rotator=rot)
maps['U'] *= -1 # CMB to IAU convention

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

def rotate_QU_frame(q,u, rot):
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

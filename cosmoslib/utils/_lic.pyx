"""Line integral convolution taken from
https://scipy-cookbook.readthedocs.io/items/LineIntegralConvolution.html

Section author: AMArchibald

"""

import numpy as np
cimport numpy as np

cdef void _advance(float vx, float vy,
        int* x, int* y, float*fx, float*fy, int w, int h):
    cdef float tx, ty
    if vx>=0:
        tx = (1-fx[0])/vx
    else:
        tx = -fx[0]/vx
    if vy>=0:
        ty = (1-fy[0])/vy
    else:
        ty = -fy[0]/vy
    if tx<ty:
        if vx>=0:
            x[0]+=1
            fx[0]=0
        else:
            x[0]-=1
            fx[0]=1
        fy[0]+=tx*vy
    else:
        if vy>=0:
            y[0]+=1
            fy[0]=0
        else:
            y[0]-=1
            fy[0]=1
        fx[0]+=ty*vx
    if x[0]>=w:
        x[0]=w-1 # FIXME: other boundary conditions?
    if x[0]<0:
        x[0]=0 # FIXME: other boundary conditions?
    if y[0]<0:
        y[0]=0 # FIXME: other boundary conditions?
    if y[0]>=h:
        y[0]=h-1 # FIXME: other boundary conditions?


#np.ndarray[float, ndim=2]
def line_integral_convolution(
        np.ndarray[float, ndim=3] vectors,
        np.ndarray[float, ndim=2] texture,
        np.ndarray[float, ndim=1] kernel):
    cdef int i,j,k,x,y
    cdef int h,w,kernellen
    cdef int t
    cdef float fx, fy, tx, ty
    cdef np.ndarray[float, ndim=2] result

    h = vectors.shape[0]
    w = vectors.shape[1]
    t = vectors.shape[2]
    kernellen = kernel.shape[0]
    if t!=2:
        raise ValueError("Vectors must have two components (not %d)" % t)
    result = np.zeros((h,w),dtype=np.float32)

    for i in range(h):
        for j in range(w):
            x = j
            y = i
            fx = 0.5
            fy = 0.5

            k = kernellen//2
            #print i, j, k, x, y
            result[i,j] += kernel[k]*texture[x,y]
            while k<kernellen-1:
                _advance(vectors[y,x,0],vectors[y,x,1],
                        &x, &y, &fx, &fy, w, h)
                k+=1
                #print i, j, k, x, y
                result[i,j] += kernel[k]*texture[x,y]

            x = j
            y = i
            fx = 0.5
            fy = 0.5

            while k>0:
                _advance(-vectors[y,x,0],-vectors[y,x,1],
                        &x, &y, &fx, &fy, w, h)
                k-=1
                #print i, j, k, x, y
                result[i,j] += kernel[k]*texture[x,y]

    return result


def lic_flow(vectors,len_pix=10):
    vectors = np.asarray(vectors)
    m,n,two = vectors.shape
    if two!=2:
        raise ValueError

    result = np.zeros((2*len_pix+1,m,n,2),dtype=np.int32) # FIXME: int16?
    center = len_pix
    result[center,:,:,0] = np.arange(m)[:,np.newaxis]
    result[center,:,:,1] = np.arange(n)[np.newaxis,:]

    for i in range(m):
        for j in range(n):
            y = i
            x = j
            fx = 0.5
            fy = 0.5
            for k in range(len_pix):
                vx, vy = vectors[y,x]
                print x, y, vx, vy
                if vx>=0:
                    tx = (1-fx)/vx
                else:
                    tx = -fx/vx
                if vy>=0:
                    ty = (1-fy)/vy
                else:
                    ty = -fy/vy
                if tx<ty:
                    print "x step"
                    if vx>0:
                        x+=1
                        fy+=vy*tx
                        fx=0.
                    else:
                        x-=1
                        fy+=vy*tx
                        fx=1.
                else:
                    print "y step"
                    if vy>0:
                        y+=1
                        fx+=vx*ty
                        fy=0.
                    else:
                        y-=1
                        fx+=vx*ty
                        fy=1.
                if x<0: x=0
                if y<0: y=0
                if x>=n: x=n-1
                if y>=m: y=m-1
                result[center+k+1,i,j,:] = y, x
    return result

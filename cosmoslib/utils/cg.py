"""try to implement a conjugate gradient method following

following: https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
"""

def CG(A, b, x0, eps=0.01, imax=50):
    """Solve linear system Ax = b, starting from x0.
    The iterative process stops when the residue falls below eps.
    Eq. 45 - 49.

    Parameters
    ----------
    A: 2D matrix
    b: column vector
    x0: start pos, column vector
    eps: minimum residue (eps * delta_0) to claim converge
    imax: maximum iteration to run

    Return
    ------
    x: column vector that solves Ax = b

    """
    i = 0
    x = x0
    # residue
    r = b - A @ x
    # step in the direction of residue
    d = r
    # initial delta^2
    delta_new = np.dot(r,r)
    delta_0 = delta_new
    while i < i_max and delta_new > eps**2 * delta_0:
        alpha = delta_new / np.einsum('i,ij,j', d,A,d)
        x += alpha * d
        # correct for floating point error at some point
        # not useful for high tolerance but good to keep
        # in mind
        if i % 50 == 0:
            r = b - A@x
        else:
            r -= alpha*q
        delta_old = delta_new
        delta_new = np.dot(r, r)
        beta = delta_new / delta_old
        d = r + beta*d
        i += 1
    return x


def PCG(A, b, x0, M_inv, eps=0.01, imax=50):
    """Solve linear system Ax = b, starting from x0.
    The iterative process stops when the residue falls below eps.

    Parameters
    ----------
    A: 2D matrix
    b: column vector
    x0: start pos, column vector
    M_inv: inv of M (precondition matrix) that approximate A
    eps: minimum residue (eps * delta_0) to claim converge
    imax: maximum iteration to run

    Return
    ------
    x: column vector that solves Ax = b

    """
    i = 0
    x = x0
    # residue
    r = b - A @ x
    # step in the direction of residue
    d = M_inv @ r
    # initial delta^2
    delta_new = np.dot(r,d)
    delta_0 = delta_new
    while i < i_max and delta_new > eps**2 * delta_0:
        alpha = delta_new / np.einsum('i,ij,j', d,A,d)
        x += alpha * d
        if i % 50 == 0:
            r = b - A@x
        else:
            r -= alpha*q
        s = M_inv @ r
        delta_old = delta_new
        delta_new = np.dot(r, s)
        beta = delta_new / delta_old
        d = s + beta*d
        i += 1
    return x

import numpy as np
from numba import njit, prange
import numba
from numba import uint32, uint64, float32, float64, int32, types
from math import floor, pi, cos, sin, sqrt
from scipy.interpolate import splprep, splev


@njit(inline="always")
def gradF_stencil_2D(F, i, j, dx, dy):
    """
    Stencil for computing the gradient of F, a grid of 2D vectors, at (i, j),
    with spacing dx and dy. Not boundary safe.

    Parameters
    ----------
    F : np.ndarray, shape = (nx, ny, 2)
        values of the vector F over grid.
    i : int
        index corresponding to the first dimension.
    j : int
        index corresponding to the second dimension.
    dx : float
        spacing in x-direction.
    dy : float
        spacing in y-direction.

    Returns
    -------
    dFxdx : float
        derivative of first component in x-direction.
    dFxdy : float
        derivative of first component in y-direction.
    dFydx : float
        derivative of second component in x-direction.
    dFydy : float
        derivative of second component in y-direction.

    """

    dFxdx = (F[i + 1, j, 0] - F[i - 1, j, 0]) / (2 * dx)
    dFxdy = (F[i, j + 1, 0] - F[i, j - 1, 0]) / (2 * dy)
    dFydx = (F[i + 1, j, 1] - F[i - 1, j, 1]) / (2 * dx)
    dFydy = (F[i, j + 1, 1] - F[i, j - 1, 1]) / (2 * dy)

    return dFxdx, dFxdy, dFydx, dFydy


@njit(inline="always")
def gradF_aux_stencil_2D(F_aux, i, j, h):
    """
    Stencil for computing the gradient of F_aux, a grid of 2D vectors, at (i, j),
    using the aux grid, with spacing h.

    Parameters
    ----------
    F_aux : np.ndarray, shape = (nx, ny, n_aux, 2)
        values of the vector F over aux-grid.
    i : int
        index corresponding to the first dimension.
    j : int
        index corresponding to the second dimension.
    h : float
        aux grid spacing.

    Returns
    -------
    dFxdx : float
        derivative of first component in x-direction.
    dFxdy : float
        derivative of first component in y-direction.
    dFydx : float
        derivative of second component in x-direction.
    dFydy : float
        derivative of second component in y-direction.

    """

    dFxdx = (F_aux[i, j, 0, 0] - F_aux[i, j, 1, 0]) / (2 * h)
    dFxdy = (F_aux[i, j, 2, 0] - F_aux[i, j, 3, 0]) / (2 * h)
    dFydx = (F_aux[i, j, 0, 1] - F_aux[i, j, 1, 1]) / (2 * h)
    dFydy = (F_aux[i, j, 2, 1] - F_aux[i, j, 3, 1]) / (2 * h)

    return dFxdx, dFxdy, dFydx, dFydy


@njit(inline="always")
def gradF_main_stencil_2D(F_aux, i, j, dx, dy):
    """
    Stencil for computing the gradient of F_aux, a grid of 2D vectors, at (i, j),
    using the main grid, with spacing dx, dy. Not boundary safe.

    Parameters
    ----------
    F_aux : np.ndarray, shape = (nx, ny, 5, 2)
        values of the vector F over aux-grid.
    i : int
        index corresponding to the first dimension.
    j : int
        index corresponding to the second dimension.
    dx : float
        spacing in x-direction.
    dy : float
        spacing in y-direction.

    Returns
    -------
    dFxdx : float
        derivative of first component in x-direction.
    dFxdy : float
        derivative of first component in y-direction.
    dFydx : float
        derivative of second component in x-direction.
    dFydy : float
        derivative of second component in y-direction.

    """

    dFxdx = (F_aux[i + 1, j, -1, 0] - F_aux[i - 1, j, -1, 0]) / (2 * dx)
    dFxdy = (F_aux[i, j + 1, -1, 0] - F_aux[i, j - 1, -1, 0]) / (2 * dy)
    dFydx = (F_aux[i + 1, j, -1, 1] - F_aux[i - 1, j, -1, 1]) / (2 * dx)
    dFydy = (F_aux[i, j + 1, -1, 1] - F_aux[i, j - 1, -1, 1]) / (2 * dy)

    return dFxdx, dFxdy, dFydx, dFydy


@njit(inline="always")
def gradUV_stencil_2D(U, V, i, j, dx, dy):
    """
    Stencil for computing the gradient of velocity, defined by U, V, at (i, j),
    with spacing dx and dy. Not boundary safe.

    Parameters
    ----------
    U : np.ndarray, shape = (nx, ny)
        velocity in x-direction.
    V : np.ndarray, shape = (nx, ny)
        velocity in y-direction.
    i : int
        index corresponding to the first dimension.
    j : int
        index corresponding to the second dimension.
    dx : float
        spacing in x-direction.
    dy : float
        spacing in y-direction.

    Returns
    -------
    dUdx : float
        derivative of x velocity in x-direction.
    dUdy : float
        derivative of x velocity in y-direction.
    dVdx : float
        derivative of y velocity in x-direction.
    dVdy : float
        derivative of y velocity in y-direction.

    """
    dUdx = (U[i + 1, j] - U[i - 1, j]) / (2 * dx)
    dUdy = (U[i, j + 1] - U[i, j - 1]) / (2 * dy)
    dVdx = (V[i + 1, j] - V[i - 1, j]) / (2 * dx)
    dVdy = (V[i, j + 1] - V[i, j - 1]) / (2 * dy)

    return dUdx, dUdy, dVdx, dVdy


@njit(inline="always")
def eigvalsh_max_2D(A):
    """
    Computes the maximum eigenvalue for a Hermetian 2x2 array A.

    Parameters
    ----------
    A : np.ndarray, shape = (2, 2)
        Hermetian 2x2 array.

    Returns
    -------
    float
        maximum eigenvalue of A.

    """

    a, b, d = A[0, 0], A[0, 1], A[1, 1]
    trace = a + d
    discriminant = sqrt((a - d) ** 2 + 4 * (b**2))

    return 0.5 * (trace + discriminant)


@njit(inline="always")
def inv_2D(A):
    """
    Computes the inverse of a 2x2 array A.

    Parameters
    ----------
    A : np.ndarray, shape = (2, 2)
        2x2 array.

    Returns
    -------
    np.ndarray, shape = (2, 2)
        inverse of A.

    """
    a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]

    det = a * d - b * c

    if det != 0:
        return np.array([[d, -b], [-c, a]]) / det
    else:
        return np.zeros((2, 2), numba.float64)


@njit(inline="always")
def vec_dot_2D(v1, v2):
    """
    Vector dot product for 2D vectors.

    Parameters
    ----------
    v1 : np.ndarray, shape=(2,)
        first vector.
    v2 : np.ndarray, shape=(2,)
        second vector.

    Returns
    -------
    float
        dot product.

    """

    return v1[0] * v2[0] + v1[1] * v2[1]


@njit(inline="always")
def vec_dot_3D(v1, v2):
    """
    Vector dot product for 3D vectors.

    Parameters
    ----------
    v1 : np.ndarray, shape=(3,)
        first vector.
    v2 : np.ndarray, shape=(3,)
        second vector.

    Returns
    -------
    float
        dot product.

    """

    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


@njit
def unravel_index(index, shape):
    """
    Numba currently does not implement np.unravel_index so we create an implementation
    here for the specific case where order='C' (row-major).

    Parameters
    ----------
    index : int
        raveled index.
    shape : np.ndarray, shape = (ndims,)
        shape of array used for unraveling index.

    Returns
    -------
    np.ndarray
        array containing the unraveled index.

    """

    shape = np.flip(shape)
    arr_ind = np.zeros(len(shape), numba.int64)
    ind = index
    for i in range(len(shape) - 1):
        div_mod = divmod(ind, shape[i])
        arr_ind[i] = div_mod[1]
        ind = div_mod[0]

    arr_ind[-1] = div_mod[0]

    return np.flip(arr_ind)


@njit
def ravel_index(inds, shape):
    """
    Finds raveled index corresponding to grid index given by inds from array with
    shape=shape where shape must be a np.ndarray.

    Parameters
    ----------
    inds : np.ndarray, shape = (ndims,)
        array containing indices to be raveled.
    shape : np.ndarray, shape = (ndims,)
        shape of array used for raveling index.

    Returns
    -------
    r_ind : int
        raveled index.

    """

    r_ind = inds[-1]
    for i, ind in enumerate(inds[:-1]):
        r_ind += ind * np.prod(shape[i + 1 :])

    return r_ind


@njit
def finite_diff_2D(f, i, j, h, axis, direction="c"):
    """
    Compute 2nd order partial finite difference in the array f @ [i,j] along axis=axis
    and using a directional difference scheme defined by direction.

    Parameters
    ----------
    f : np.ndarray, shape = (nx,ny)
        array on which finite differencing is performed.
    i : int
        index for axis=0 at which finite differencing is performed.
    j : int
        index for axis=1 at which finite differencing is performed.
    h : float
        spacing in axis direction.
    axis : int
        axis over which finite differencing is performed.
    direction : str, optional
        finite differencing direction, optional values are 'b' for backward,
        'c' for centered, and 'f' for forward. The default is 'c'.

    Returns
    -------
    df : float
        finite difference value.

    """

    if axis == 0:
        if direction == "c":
            df = (f[i + 1, j] - f[i - 1, j]) / (2 * h)
        elif direction == "b":
            df = (3 * f[i, j] - 4 * f[i - 1, j] + f[i - 2, j]) / (2 * h)
        elif direction == "f":
            df = (-f[i + 2, j] + 4 * f[i + 1, j] - 3 * f[i, j]) / (2 * h)
        else:
            print("Valid difference directions are 'c', 'b', and 'f'")
    if axis == 1:
        if direction == "c":
            df = (f[i, j + 1] - f[i, j - 1]) / (2 * h)
        elif direction == "b":
            df = (3 * f[i, j] - 4 * f[i, j - 1] + f[i, j - 2]) / (2 * h)
        elif direction == "f":
            df = (-f[i, j + 2] + 4 * f[i, j + 1] - 3 * f[i, j]) / (2 * h)
        else:
            print("Valid difference directions are 'c', 'b', and 'f'")

    return df


@njit
def finite_diff_ND(f, ind, h, axis, shape, direction=0):
    """
    Compute 2nd order partial finite difference in the array f @ [ind] where f is an
    (n_1*n_2*...*n_ndims) array. Axis=axis determines the axis the finite differencing
    will be performed along shape is [n1,n2,...,n_ndims]. shape must be a np.array and
    'ij' indexing is assumed.

    Parameters
    ----------
    f : np.ndarray, shape = (nx_1*nx_2*...*nx_ndims,)
        flattened array of which finite differencing is performed.
    ind : int
        raveled index at which finite differencing is performed.
    h : float
        spacing in axis direction.
    axis : int
        axis over which finite differencing is performed.
    shape : np.ndarray, shape = (ndims,)
        shape of original array before raveled.
    direction : str, optional
        finite differencing direction, optional values are -1 for backward,
        0 for centered, and 1 for forward. The default is 0.

    Returns
    -------
    df : float
        finite difference value.

    """

    di = np.zeros(len(ind), numba.int64)
    di[axis] = 1
    if direction == 0:
        i_p1 = ravel_index(ind + di, shape)
        i_m1 = ravel_index(ind - di, shape)
        df = (f[i_p1] - f[i_m1]) / (2 * h)
    elif direction == -1:
        i = ravel_index(ind, shape)
        i_m1 = ravel_index(ind - di, shape)
        i_m2 = ravel_index(ind - 2 * di, shape)
        df = (3 * f[i] - 4 * f[i_m1] + f[i_m2]) / (2 * h)
    elif direction == 1:
        i = ravel_index(ind, shape)
        i_p1 = ravel_index(ind + di, shape)
        i_p2 = ravel_index(ind + 2 * di, shape)
        df = (-f[i_p2] + 4 * f[i_p1] - 3 * f[i]) / (2 * h)
    else:
        print("Valid difference directions are 0: centered, -1: backward, and 1: forward")

    return df


@njit
def finite_diff_2D_2nd(f, i, j, h, axis, direction="c"):
    """
    Compute 2nd order partial finite difference in the array f @ [i,j] along axis=axis and using a
    directional difference scheme defined by direction for the second derivative. axis=2 is for the
    mixed partial finite difference.
    """
    if axis == 0:
        if direction == "c":
            df = (f[i + 1, j] - 2 * f[i, j] + f[i - 1, j]) / (h**2)
        elif direction == "b":
            df = (2 * f[i, j] - 5 * f[i - 1, j] + 4 * f[i - 2, j] - f[i - 3, j]) / (h**3)
        elif direction == "f":
            df = (2 * f[i, j] - 5 * f[i + 1, j] + 4 * f[i + 2, j] - f[i + 3, j]) / (h**3)
        else:
            print("Valid difference directions are 'c', 'b', and 'f'")
    if axis == 1:
        if direction == "c":
            df = (f[i, j + 1] - 2 * f[i, j] + f[i, j - 1]) / (h**2)
        elif direction == "b":
            df = (2 * f[i, j] - 5 * f[i, j - 1] + 4 * f[i, j - 2] - f[i, j - 3]) / (h**3)
        elif direction == "f":
            df = (2 * f[i, j] - 5 * f[i, j + 1] + 4 * f[i, j + 2] - f[i, j + 3]) / (h**3)
        else:
            print("Valid difference directions are 'c', 'b', and 'f'")
    if axis == 2:
        if direction == "c":
            df = (f[i + 1, j + 1] - f[i + 1, j - 1] - f[i - 1, j + 1] + f[i - 1, j - 1]) / (
                4 * h**2
            )
        else:
            print("Valid difference directions are 'c' for mixed derivative")
    return df


@njit(parallel=True)
def curl_vel(u, v, dx, dy):
    """
    Compute curl of vector field defined by u and v.

    Parameters
    ----------
    u : np.ndarray, shape = (nx,ny)
        array containing x component of vector field.
    v : np.ndarray, shape = (nx,ny)
        array containing y component of vector field.
    dx : float
        spacing in grid in x-direction.
    dy : float
        spacing in grid in y-direction.

    Returns
    -------
    curl : np.ndarray, shape = (nx,ny)
        array containing values of curl of vector field defined by u and v.

    """
    nx, ny = u.shape
    curl = np.zeros((nx, ny), numba.float64)
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            dfydx = (v[i + 1, j] - v[i - 1, j]) / (2 * dx)
            dfxdy = (u[i, j + 1] - u[i, j - 1]) / (2 * dy)

            curl[i, j] = dfydx - dfxdy

    return curl


@njit(parallel=True)
def curl_vel_tspan(u, v, dx, dy):
    """
    Compute curl of vector field defined by u and v over some timespan.

    Parameters
    ----------
    u : np.ndarray, shape = (nt,nx,ny)
        array containing x component of vector field.
    v : np.ndarray, shape = (nt,nx,ny)
        array containing y component of vector field.
    dx : float
        spacing in grid in x-direction.
    dy : float
        spacing in grid in y-direction.

    Returns
    -------
    curl : np.ndarray, shape = (nt,nx,ny)
        array containing values of curl of vector field defined by u and v.

    """
    nt, nx, ny = u.shape
    curl = np.zeros((nt, nx, ny), numba.float64)
    for k in prange(nt):
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                dfydx = (v[k, i + 1, j] - v[k, i - 1, j]) / (2 * dx)
                dfxdy = (u[k, i, j + 1] - u[k, i, j - 1]) / (2 * dy)

                curl[k, i, j] = dfydx - dfxdy

    return curl


@njit(parallel=True)
def curl_func(fnc, x, y, h=1e-3):
    """
    Compute curl over x,y of vector field defined by fnc.

    Parameters
    ----------
    fnc : jit-callable
        callable containing returing x and y components of vector field.
    x : np.ndarray, shape = (nx,)
        array containing x-values.
    y : np.ndarray, shape = (ny,)
        array containing y-values.
    h : float, optional
        spacing used in finite differencing. The default is 1e-3.

    Returns
    -------
    curlf : np.ndarray, shape = (nx,ny)
        array containing values of curl of f.

    """

    nx = len(x)
    ny = len(y)
    curlf = np.zeros((nx, ny), numba.float64)
    dx_vec = np.array([h, 0.0], numba.float64)
    dy_vec = np.array([0.0, h], numba.float64)
    for i in prange(nx):
        for j in range(ny):
            pt = np.array([x[i], y[j]])

            dfydx = (fnc(pt + dx_vec)[1] - fnc(pt - dx_vec)[1]) / (2 * h)
            dfxdy = (fnc(pt + dy_vec)[0] - fnc(pt - dy_vec)[0]) / (2 * h)

            curlf[i, j] = dfydx - dfxdy

    return curlf


@njit(parallel=True)
def curl_func_tspan(fnc, t, x, y, h=1e-3):
    """
    Compute curl over x,y of vector field defined by func over times t.

    Parameters
    ----------
    fnc : jit-callable
        callable containing returing x and y components of vector field.
    x : np.ndarray, shape = (nx,)
        array containing x-values.
    y : np.ndarray, shape = (ny,)
        array containing y-values.
    h : float, optional
        spacing used in finite differencing. The default is 1e-3.

    Returns
    -------
    curlf : np.ndarray, shape = (nx,ny)
        array containing values of curl of f.

    """
    nt = len(t)
    nx = len(x)
    ny = len(y)
    curlf = np.zeros((nt, nx, ny), numba.float64)
    dx_vec = np.array([0.0, h, 0.0], numba.float64)
    dy_vec = np.array([0.0, 0.0, h], numba.float64)
    for k in prange(nt):
        for i in range(nx):
            for j in range(ny):
                pt = np.array([t[k], x[i], y[j]])

                dfydx = (fnc(pt + dx_vec)[1] - fnc(pt - dx_vec)[1]) / (2 * h)
                dfxdy = (fnc(pt + dy_vec)[0] - fnc(pt - dy_vec)[0]) / (2 * h)

                curlf[k, i, j] = dfydx - dfxdy

    return curlf


@njit
def composite_simpsons(f, h):
    """
    Composite Simpson's 1/3 rule to compute integral of f between endpoitns of pts with
    regular spacing given by h.

    Parameters
    ----------
    f : np.ndarray, shape = (n+1,)
        values of f at regularly spaced points.
    h : float
        value of spacing between points at which f was evaluated.

    Returns
    -------
    val : float
        value of integral.

    """
    n = len(f) - 1
    if n % 2 == 0:
        val = f[0]
        val += f[n]
        for k in range(1, n):
            if k % 2 != 0:
                val += 4 * f[k]
            else:
                val += 2 * f[k]

        val *= h / 3

    else:
        n -= 1
        val = f[0]
        val += f[-2]
        for k in range(1, n):
            if k % 2 != 0:
                val += 4 * f[k]
            else:
                val += 2 * f[k]

        val *= h / 3
        val += (5 * h / 12) * f[-1] + (2 * h / 3) * f[-2] - (h / 12) * f[-3]

    return val


@njit
def composite_simpsons_38_irregular(f, h):
    """
    Composite Simpson's 3/8 rule to compute integral of f between endpoitns of pts with
    irregular spacing given by h which is an np.ndarray.

    Parameters
    ----------
    f : np.ndarray, shape = (n+1,)
        values of f at irregularly spaced points.
    h : np.ndarray, shape = (n,)
        values of spacing between points at which f was evaluated.

    Returns
    -------
    val : float
        value of integral.

    """

    n = len(h)
    val = 0
    for k in range(int(n / 2)):
        h0 = h[2 * k]
        h1 = h[2 * k + 1]
        val += (
            (1 / 6)
            * (h0 + h1)
            * (
                f[2 * k] * (2 - h1 / h0)
                + f[2 * k + 1] * ((h0 + h1) ** 2) / (h0 * h1)
                + f[2 * k + 2] * (2 - h0 / h1)
            )
        )

    if n % 2 == 1:
        h1 = h[n - 1]
        h2 = h[n - 2]
        alph = (2 * h1**2 + 3 * h1 * h2) / (6 * (h2 + h1))
        beta = (h1**2 + 3 * h1 * h2) / (6 * h2)
        eta = (h1**3) / (6 * h2 * (h2 + h1))

        val += alph * f[-1] + beta * f[-2] - eta * f[-3]

    return val


@njit
def dist_2d(p1, p2):
    """
    Compute 2D Euclidean distance between p1 and p2.

    Parameters
    ----------
    p1 : np,ndarry, shape = (2,)
        point 1.
    p2 : np,ndarry, shape = (2,)
        point 2.

    Returns
    -------
    float
        distance between p1 and p2.

    """

    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


@njit
def dist_tol(point, arr, tol):
    """
    Check if pt is within tolerance of any point in arr.

    Parameters
    ----------
    point : np.ndarray, shape = (2,)
        point.
    arr : np.ndarray, shape = (n,2)
        array containing points.
    tol : float
        tolerance used for checking distance.

    Returns
    -------
    near : boolean
        truth value determining if point is within tol of any point in arr.

    """
    k = 0
    dist = np.zeros(len(arr), numba.float64)
    near = False
    for pt in arr:
        dist[k] = dist_2d(pt, point)
        if dist[k] < tol:
            near = True
            break
        k += 1
    return near


@njit
def shoelace(polygon):
    """
    Compute area of simple polygon using shoelace algorithm.

    Parameters
    ----------
    polygon : np.ndarray, shape = (n,2)
        array containing vertices of polygon, first and last vertex should be the same.

    Returns
    -------
    area : float
        area of polygon.

    """

    n = polygon.shape[0]
    x0 = polygon[0, 0]
    y1 = polygon[1, 1]
    area = x0 * y1
    for k in range(1, n - 1):
        x1 = polygon[k, 0]
        y2 = polygon[k + 1, 1]
        y0 = polygon[k - 1, 1]
        area += x1 * (y2 - y0)

    area = abs(0.5 * (area - polygon[n - 1, 0] * polygon[n - 2, 1]))

    return area


@njit
def max_in_radius(arr, r, dx, dy, n=-1, min_val=0.0):
    """
    Finds n local maxima values in arr such that each max is a local maximum within radius r where
    spacing in arr is given by dx,dy. If all local maxima are desired, set n = -1. Should pass a
    copy of arr to the function to avoid orginial arr being overwritten (i.e. pass in arr.copy()).

    Parameters
    ----------
    arr : np.ndarray, shape = (nx,ny)
        array in which local maxima are to be found.
    r : float
        radius in which points will be discared after a maximum is found at the center.
    dx : float
        gird spacing in x-direction.
    dy : float
        grid spacing in y-direction.
    n : int, optional
        number of maxima to return, -1 returns all. The default is -1.
    min_val : float, optional
        miniumum value allowed for local maxima. The default is 0.0.

    Returns
    -------
    max_vals : np.ndarray, shape = (k,) where k number of maxima found
        maxima values in radius.
    max_inds : np.ndarray, shape = (k,) where k number of maxima found
        indices corresponding to max_vals.

    """

    nx, ny = arr.shape
    ix = floor(r / dx)
    iy = floor(r / dy)
    arr_shape = np.array([nx, ny])
    if n == -1:
        max_inds = np.zeros((int(nx * ny / 2), 2), numba.int32)
        max_vals = np.zeros(int(nx * ny / 2), numba.float64)
        k = 0
        while np.max(arr) > min_val:
            max_ind = np.argmax(arr)
            max_inds[k, :] = unravel_index(max_ind, arr_shape)
            max_vals[k] = arr[max_inds[k, 0], max_inds[k, 1]]
            max_vals[k] = arr[max_inds[k, 0], max_inds[k, 1]]
            arr[
                max(0, max_inds[k, 0] - ix) : min(nx, max_inds[k, 0] + ix),
                max(0, max_inds[k, 1] - iy) : min(ny, max_inds[k, 1] + iy),
            ] = 0
            k += 1

    else:
        max_inds = np.zeros((n, 2), numba.int32)
        max_vals = np.zeros(n, numba.float64)
        k = 0
        while np.max(arr) > min_val and k < n:
            max_ind = np.argmax(arr)
            max_inds[k, :] = unravel_index(max_ind, arr_shape)
            max_vals[k] = arr[max_inds[k, 0], max_inds[k, 1]]
            arr[
                max(0, max_inds[k, 0] - ix) : min(nx, max_inds[k, 0] + ix),
                max(0, max_inds[k, 1] - iy) : min(ny, max_inds[k, 1] + iy),
            ] = 0
            k += 1

    return max_vals[:k], max_inds[:k, :]


@njit
def gen_circ(r, c, n, xlims=None, ylims=None):
    """
    Generate n points on a circle with radius r and center c.

    Parameters
    ----------
    r : float
        radius of circle.
    c : np.ndarray, shape = (2,)
        center of circle.
    n : int
        number of points on circle.
    xlims : tuple, optional
        boundary in x-direction, points outside of this boundary will not be returned.
        The default is None.
    ylims : tuple, optional
        boundary in y-direction, points outside of this boundary will not be returned.
        The default is None.

    Returns
    -------
    pts : np.ndarray, shape = (n,2)
          points on the circle.

    """

    theta = np.linspace(0, 2 * pi, n)
    pts = np.zeros((n, 2), np.float64)
    cx = c[0]
    cy = c[1]
    for k in prange(n):
        pts[k, 0] = r * cos(theta[k]) + cx
        pts[k, 1] = r * sin(theta[k]) + cy

    if xlims is not None:
        xm = pts[:, 0] < xlims[0]
        xM = pts[:, 0] > xlims[1]
        maskx = ~np.logical_or(xm, xM)
        pts = pts[maskx, :]
    if ylims is not None:
        ym = pts[:, 0] < ylims[0]
        yM = pts[:, 0] > ylims[1]
        masky = ~np.logical_or(ym, yM)
        pts = pts[masky, :]

    return pts


@njit
def gen_filled_circ(r, n, alpha=3.0, c=np.array([0.0, 0.0]), xlims=None, ylims=None):
    """
    Generate points filling a circle with radius r and center c. Uses the sunflower
    seed arangement.

    Parameters
    ----------
    r : float
        radius of circle.
    n : int
        number of points to fill the circle.
    alpha : float, optional
        parameter determining how smooth the boundary is. The default is 3.0.
    c : np.ndarray, shape = (2,), optional
        center of the circle. The default is np.array([0.0,0.0]).
    xlims : tuple, optional
        boundary in x-direction, points outside of this boundary will not be returned.
        The default is None.
    ylims : tuple, optional
        boundary in y-direction, points outside of this boundary will not be returned.
        The default is None.

    Returns
    -------
    pts : np.ndarray, shape = (n,2)
        array containing points which fill the circle.

    """
    phi = 0.5 * (1 + 5**0.5)
    cd = 1 / (phi**2)
    ar = round(alpha * n**0.5)
    x = np.zeros(n, np.float64)
    y = np.zeros(n, np.float64)
    for k in range(1, n + 1):
        theta = 2 * pi * k * cd
        if k > n - ar:
            radius = r
        else:
            radius = r * ((k - 0.5) ** 0.5) / (n - (ar + 1) / 2) ** 0.5

        x[k - 1] = radius * cos(theta) + c[0]
        y[k - 1] = radius * sin(theta) + c[1]

    if xlims is not None:
        xm = x < xlims[0]
        xM = x > xlims[1]
        maskx = ~np.logical_or(xm, xM)
        x = x[maskx]
        y = y[maskx]
    if ylims is not None:
        ym = y < ylims[0]
        yM = y > ylims[1]
        masky = ~np.logical_or(ym, yM)
        x = x[masky]
        y = y[masky]

    pts = np.column_stack((x, y))
    return pts


@njit
def gen_filled_circ_radius(r, n, alpha=3.0, c=np.array([0.0, 0.0]), xlims=None, ylims=None):
    """
    Generate points filling a circle with radius r and center c. Uses the sunflower
    seed arangement. Also returns radius of each point from center c.

    Parameters
    ----------
    r : float
        radius of circle.
    n : int
        number of points to fill the circle.
    alpha : float, optional
        parameter determining how smooth the boundary is. The default is 3.0.
    c : np.ndarray, shape = (2,), optional
        center of the circle. The default is np.array([0.0,0.0]).
    xlims : tuple, optional
        boundary in x-direction, points outside of this boundary will not be returned.
        The default is None.
    ylims : tuple, optional
        boundary in y-direction, points outside of this boundary will not be returned.
        The default is None.

    Returns
    -------
    pts : np.ndarray, shape = (n,2)
        array containing points which fill the circle.
    radius : np.ndarray, shape = (n,), optional
        array containing radius of each point from center c.

    """
    phi = 0.5 * (1 + 5**0.5)
    cd = 1 / (phi**2)
    ar = round(alpha * n**0.5)
    x = np.zeros(n, np.float64)
    y = np.zeros(n, np.float64)
    radius = np.zeros(n, np.float64)
    for k in range(1, n + 1):
        theta = 2 * pi * k * cd
        if k > n - ar:
            radius[k - 1] = r
        else:
            radius[k - 1] = r * ((k - 0.5) ** 0.5) / (n - (ar + 1) / 2) ** 0.5

        x[k - 1] = radius[k - 1] * cos(theta) + c[0]
        y[k - 1] = radius[k - 1] * sin(theta) + c[1]

    if xlims is not None:
        xm = x < xlims[0]
        xM = x > xlims[1]
        maskx = ~np.logical_or(xm, xM)
        x = x[maskx]
        y = y[maskx]
        radius = radius[maskx]
    if ylims is not None:
        ym = y < ylims[0]
        yM = y > ylims[1]
        masky = ~np.logical_or(ym, yM)
        x = x[masky]
        y = y[masky]
        radius = radius[masky]

    pts = np.column_stack((x, y))
    return pts, radius


@njit
def arclength(pts):
    """
    Compute total arclength of curve defined by pts.

    Parameters
    ----------
    pts : np.ndarray, shape=(npts,)
        points representing curve for which arclength is to be computed.

    Returns
    -------
    arclength_ : float
        arclength of curve defined by points.

    """
    npts = len(pts)
    arclength_ = 0
    for k in prange(npts - 1):
        p0 = pts[k, :]
        p1 = pts[k + 1, :]
        arclength_ += ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5

    return arclength_


@njit
def arclength_along_arc(pts):
    """
    Compute cummulative arclength at each point defining a curve.

    Parameters
    ----------
    pts : np.ndarray, shape=(npts,)
        points representing curve for which arclength is to be computed.

    Returns
    -------
    arclength_ : np.ndarray, shape = (npts,)
        array containing cummulative arclength of curve defined by pts.

    """
    npts = len(pts)
    arclength_ = np.zeros(npts, numba.float64)
    arclength_[0] = 0.0
    for k in range(1, npts):
        p0 = pts[k - 1, :]
        p1 = pts[k, :]
        arclength_[k] = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5 + arclength_[k - 1]

    return arclength_


def interp_curve(curve, n, s=0, k=3, per=0):
    """
    Return n interpolated values of curve.

    Parameters
    ----------
    curve : np.ndarray, shape = (npts,2)
        array containing x,y-coordinates of curve being interpolated.
    n : int
        number of equally spaced interpolated points to return.
    s : float, optional
        smoothing parameter for spline, 0 will fit all data points exactly. The default is 0.
    k : int, optional
        degree of spline, using even numbers is not recommended, must be 1 <= k <= 5.
        The default is 3.
    per : int, optional
        if nonzero, data points are considered periodic. The default is 0.

    Returns
    -------
    curvei : np.ndarray, shape = (n,2)
        array containing interpolated values of curve.


    """

    tck, u = splprep([curve[:, 0], curve[:, 1]], k=k, s=s, per=per)
    xi, yi = splev(np.linspace(0, 1, n), tck)

    curvei = np.column_stack((xi, yi))
    return curvei


@njit
def wn_pt_in_poly(polygon, point):
    """
    Winding number algorithm to determine if point is inside polygon. Based off of
    java implementation - https://observablehq.com/@jrus/winding-number - by Jacob Rus.

    Parameters
    ----------
    polygon : np.ndarray, shape = (n,2)
        array containing vertices of polygon, first and last points should match.
    point : np.ndarray, shape = (2,)
        array containing the coordinates of point to be checked.

    Returns
    -------
    wn : int
        winding number, returns 1 if point is inside polygon and 0 if not.

    """
    n = len(polygon)
    ptx = point[0]
    pty = point[1]
    wn = 0
    dx1 = polygon[0, 0] - ptx
    dy1 = polygon[0, 1] - pty
    below1 = dy1 <= 0
    for k in range(1, n):
        dx0 = dx1
        dy0 = dy1
        below0 = below1
        dx1 = polygon[k, 0] - ptx
        dy1 = polygon[k, 1] - pty
        below1 = dy1 <= 0
        is_left = dx0 * dy1 - dx1 * dy0 > 0
        wn += (below0 & (below1 ^ 1) & is_left) - (below1 & (below0 ^ 1) & ~is_left)

    return wn


@njit
def pts_in_poly(polygon, pts):
    """
    Checks if any point from pts is inside polygon. If a point is, the index of the
    first point found inside polygon is returned. Else, -1 is returned.

    Parameters
    ----------
    polygon : np.ndarray, shape = (n,2)
        array containing vertices of polygon, first and last points should match.
    pts : np.ndarray, shape = (npts,2)
        array containing the coordinates of the points to be checked.

    Returns
    -------
    int
        if point from pts is found inside polygon, its index is returned,
        if not, -1 is returned.

    """

    for k in range(len(pts)):
        pt = pts[k, :]

        if wn_pt_in_poly(polygon, pt):
            return k

    return -1


@njit
def pts_in_poly_mask(polygon, pts):
    """
    Checks which points from pts are inside polygon. Returns a boolean mask
    corresponding to points inside polygon.

    Parameters
    ----------
    polygon : np.ndarray, shape = (n,2)
        array containing vertices of polygon, first and last points should match.
    pts : np.ndarray, shape = (npts,2)
        array containing the coordinates of the points to be checked.

    Returns
    -------
    mask : np.ndarray, shape = (npts,)
        bool mask with indices matching those of pts, True if in polygon, False if not.

    """
    npts = len(pts)
    mask = np.zeros((npts,), np.bool_)
    for k in range(npts):
        mask[k] = wn_pt_in_poly(polygon, pts[k, :])

    return mask


@njit(parallel=True)
def cart_prod(vecs):
    """
    Computes the Cartesian product using vectors from vecs, works for any
    number of vectors.

    Parameters
    ----------
    vecs : tuple
        tuple containing vectors representing the sets which will be used
        to compute Cartestian product, all vectors must be 1D np.ndarrays
        of the same type.

    Returns
    -------
    prod : np.ndarray, shape = (npts,nvecs)
        array containing the Cartesian product.

    """

    nvecs = len(vecs)
    dtype = vecs[0].dtype
    shape = np.zeros(nvecs, np.int32)
    for k in range(nvecs):
        shape[k] = len(vecs[k])
    npts = np.prod(shape)
    prod = np.zeros((npts, nvecs), dtype)

    if nvecs > 2:
        for k in prange(nvecs - 1):
            cl = np.prod(shape[-1:k:-1])
            arr = vecs[k]
            shapek = shape[k]
            for j in range(shapek):
                prod[j * cl : (j + 1) * cl, k] = arr[j]

            full_len = shapek * cl
            for i in range(np.prod(shape[:k]) - 1):
                prod[(i + 1) * full_len : (i + 2) * full_len, k] = prod[:full_len, k]
    else:
        cl = shape[1]
        arr = vecs[0]
        for j in prange(shape[0]):
            prod[j * cl : (j + 1) * cl, 0] = arr[j]

    nlast = shape[-1]
    arr = vecs[-1]
    for j in prange(np.prod(shape[:-1])):
        prod[j * nlast : (j + 1) * nlast, -1] = arr

    return prod


@njit
def _normalize_batch(v_arr, r=1.0, dtype=float32):
    """Normalize array of vectors."""

    norms = np.sqrt(np.sum(v_arr * v_arr, axis=1)).reshape(-1, 1)
    return (v_arr * (r / norms)).astype(dtype)


@njit
def _get_edge_key(i1, i2):
    """Generates a unique 64-bit integer key for an edge."""
    if i1 < i2:
        return (uint64(i1) << 32) | uint64(i2)
    else:
        return (uint64(i2) << 32) | uint64(i1)


# define signature so no numba warning about unsafe cast
_midpoint_sig = types.Tuple((types.uint32, types.uint32))(
    types.uint32,  # i1
    types.uint32,  # i2
    types.float32[:, :],  # verts
    types.DictType(types.uint64, types.uint32),  # cache
    types.uint32,  # vi
)


@njit(_midpoint_sig)
def _get_midpoint_index(i1, i2, verts, cache, vi):
    """
    Returns index of midpoint between verts and next index, updates
    verts array and cache if new midpoint inserted.

    """

    key = _get_edge_key(i1, i2)

    # check if midpoint already exists
    # if True, return its index and next vertex index
    if key in cache:
        return cache[key], vi

    # if not, calculate midpoint, store it, and return its index and next vertex index
    v1, v2 = verts[i1], verts[i2]
    mid = 0.5 * (v1 + v2)
    verts[vi] = mid
    cache[key] = vi

    return vi, vi + uint32(1)


@njit
def _subdivide_normalize(verts, faces, vi, cache, r):
    """
    Subdivides icosphere defined by verts and corresponding faces,
    normalizes at each step.

    """
    n_faces = len(faces)
    new_faces = np.zeros((n_faces * 4, 3), uint32)

    fi = 0
    for i in range(n_faces):
        # get and set new vertices
        i0, i1, i2 = faces[i]
        m0, vi = _get_midpoint_index(i0, i1, verts, cache, vi)
        m1, vi = _get_midpoint_index(i1, i2, verts, cache, vi)
        m2, vi = _get_midpoint_index(i2, i0, verts, cache, vi)

        # set new face indices
        new_faces[fi, :] = (i0, m0, m2)
        new_faces[fi + 1] = (i1, m0, m1)
        new_faces[fi + 2] = (i2, m1, m2)
        new_faces[fi + 3] = (m0, m1, m2)
        fi += 4

    # normalize verts to be on sphere of radius r
    verts[:vi] = _normalize_batch(verts[:vi], r, dtype=verts.dtype)

    return new_faces, vi


@njit
def _subdivide(verts, faces, vi, cache):
    """
    Subdivides icosphere defined by verts and corresponding faces,
    does not normalize to surface of sphere.
    """
    n_faces = len(faces)
    new_faces = np.zeros((n_faces * 4, 3), uint32)

    fi = 0
    for i in range(n_faces):
        # get and set new vertices
        i0, i1, i2 = faces[i]
        m0, vi = _get_midpoint_index(i0, i1, verts, cache, vi)
        m1, vi = _get_midpoint_index(i1, i2, verts, cache, vi)
        m2, vi = _get_midpoint_index(i2, i0, verts, cache, vi)

        # set new face indices
        new_faces[fi, :] = (i0, m0, m2)
        new_faces[fi + 1] = (i1, m0, m1)
        new_faces[fi + 2] = (i2, m1, m2)
        new_faces[fi + 3] = (m0, m1, m2)
        fi += 4

    return new_faces, vi


@njit
def icosphere(subdivisions, r=1.0, dtype=float32, normalize_once=False):
    """
    Generate an icosphere using subdivisions=subdivisions, with radius r.
    If normalize_once is True, normalizing to the surface of the sphere will
    only happen at the very end, if not, normalization happens at every
    level of subdivision. The former is faster, the latter will produce
    more evenly spaced points.

    Parameters
    ----------
    subdivisions : int
        number of subdivisions.
    r : float
        radius of sphere that the mesh will be defined on.
    dtype : numba.types or np.type, optional
        dtype used for the coordinates of the mesh, should be float32 or float64.
        The default is float32.
    normalize_once : bool, optional
        flag determining how normalization is handled.
        If normalize_once is True, normalizing to the surface of the sphere will
        only happen at the very end, if not, normalization happens at every
        level of subdivision. The former is faster, the latter will produce
        more evenly spaced points. The default is False.

    Returns
    -------
    verts : np.ndarray, shape=(10 * 4**subdivisions + 2, 3)
        array containing vertices of mesh.
    faces : np.ndarray, shape=(20 * 4**subdivisions, 3)
        array containing faces of mesh.

    """

    # golden ratio
    phi = (1.0 + sqrt(5.0)) / 2.0

    # the 12 vertices of the initial icosahedron
    verts0 = np.array(
        [
            [-1.0, phi, 0.0],
            [1.0, phi, 0.0],
            [-1.0, -phi, 0.0],
            [1.0, -phi, 0.0],
            [0.0, -1.0, phi],
            [0.0, 1.0, phi],
            [0.0, -1.0, -phi],
            [0.0, 1.0, -phi],
            [phi, 0.0, -1.0],
            [phi, 0.0, 1.0],
            [-phi, 0.0, -1.0],
            [-phi, 0.0, 1.0],
        ],
        float32,
    )

    r = float32(r)
    # normalize the vertices to lie on the sphere of radius r
    verts0 = _normalize_batch(verts0, r=r)

    # the 20 faces of the initial icosahedron
    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        uint32,
    )

    # preallocate verts array
    verts = np.zeros((10 * 4**subdivisions + 2, 3), dtype=dtype)
    vi = uint32(12)
    verts[:vi] = verts0

    # initialize edge cache
    cache = numba.typed.Dict.empty(key_type=uint64, value_type=uint32)
    if normalize_once:
        for _ in range(subdivisions):
            faces, vi = _subdivide(verts, faces, vi, cache)

        verts = _normalize_batch(verts, r=r, dtype=dtype)

        return verts, faces

    else:
        for _ in range(subdivisions):
            faces, vi = _subdivide_normalize(verts, faces, vi, cache, r)

        return verts.astype(dtype), faces


@njit
def _add_neighbor_pair(i1, i2, neighbors, counts):
    """Helper to add a neighbor relationship, avoiding duplicates."""

    # add i2 to i1's neighbor list
    # check if i2 is already a neighbor of i1
    if i2 not in neighbors[i1, : counts[i1]]:
        neighbors[i1, counts[i1]] = i2
        counts[i1] += 1

    # add i1 to i2's neighbor list
    # check if i1 is already a neighbor of i2
    if i1 not in neighbors[i2, : counts[i2]]:
        neighbors[i2, counts[i2]] = i1
        counts[i2] += 1


@njit
def find_neighbors(faces, num_verts):
    """Finds all neighbors for each vertex."""

    # preallocate neighbors and counts, vertices with 5 neighbors will have -1
    # in last entry
    neighbors = np.full((num_verts, 6), -1, int32)
    counts = np.zeros(num_verts, int32)

    for i in range(len(faces)):
        i0, i1, i2 = faces[i]
        _add_neighbor_pair(i0, i1, neighbors, counts)
        _add_neighbor_pair(i1, i2, neighbors, counts)
        _add_neighbor_pair(i2, i0, neighbors, counts)

    return neighbors


def convert_vel_to_3D(u, v, lon, lat, deg2rad=False, pole="both"):
    """
    Convert velocity from lon-lat directions to xyz directions.

    Parameters
    ----------
    u : np.ndarray, shape=(nt, nx, ny)
        velocity in the local east direction.
    v : np.ndarray, shape=(nt, nx, ny)
        velocity in the local north direction.
    lon : np.ndarray, shape=(nx,)
        array containing longitude values.
    lat : np.ndarray, shape=(ny,)
        array containing latitude values.
    deg2rad : bool, optional
        flag to determine if lon-lat need to be converted to radians. The default is False.
    pole : str, optional
        str to determine if any poles are included in the velocity fields. Options are
        "both", "north", or "south". If pole is anything else, treated as no poles. The default is "both".

    Returns
    -------
    vx : np.ndarray, shape=(nt, nx, ny)
        velocity in the x direction.
    vy : np.ndarray, shape=(nt, nx, ny)
        velocity in the y direction.
    vz : np.ndarray, shape=(nt, nx, ny)
        velocity in the z direction.

    """
    nt, nlon, nlat = u.shape
    if deg2rad:
        lon = np.deg2rad(lon)
        lat = np.deg2rad(lat)

    Lon_rad, Lat_rad = np.meshgrid(lon, lat, indexing="ij")

    sinLon = np.sin(Lon_rad)
    sinLat = np.sin(Lat_rad)
    cosLon = np.cos(Lon_rad)
    cosLat = np.cos(Lat_rad)

    vx = -u * sinLon - v * cosLon * sinLat
    vy = u * cosLon - v * sinLon * sinLat
    vz = v * cosLat

    # average all velocity vectors at the poles, if poles are included
    pole = pole.lower()
    if pole == "both" or pole == "south":
        vx[:, :, 0] = np.mean(vx[:, :, 0], axis=1, keepdims=True)
        vy[:, :, 0] = np.mean(vy[:, :, 0], axis=1, keepdims=True)
        vz[:, :, 0] = np.mean(vz[:, :, 0], axis=1, keepdims=True)

    if pole == "both" or pole == "north":
        vx[:, :, -1] = np.mean(vx[:, :, -1], axis=1, keepdims=True)
        vy[:, :, -1] = np.mean(vy[:, :, -1], axis=1, keepdims=True)
        vz[:, :, -1] = np.mean(vz[:, :, -1], axis=1, keepdims=True)

    return (vx, vy, vz)


@njit(parallel=True)
def displacements_proj_ico(points, e1, e2, neighbors, mask=None):
    """
    Compute displacement array and project onto local coordinates.

    Parameters
    ----------
    points : np.ndarray, shape=(nx, ny, 3)
        collection of (x, y, z) points.
    e1 : np.ndarray, shape = (nx, ny, 2)
        local "x" basis vector at each point.
    e2 : np.ndarray, shape = (nx, ny, 2)
        local "y" basis vector at each point.
    mask : None or np.ndarray, shape = (nx, ny), optional
        for masked data, pass in a boolean mask corresponding to nan values
        (True indicates a nan value). To avoid erroneous computations at mask
        boundaries, mask passed in should be dilated using the
        binary_mask_dilation_mesh function from the utils module. The default is None.

    Returns
    -------
    X : np.ndarray, shap=(nx, ny, 2, n)
        array corresponding to displacements of n neighbors for each point.

    """
    FIRST_POINTS = 12
    npts = len(points)
    nhbrs = 6
    X = np.zeros((npts, 2, nhbrs), float64)

    for i in range(FIRST_POINTS):
        if mask is None or not mask[i]:
            p0 = points[i, :]
            e1i = e1[i, :]
            e2i = e2[i, :]
            for k in range(nhbrs - 1):
                ni = neighbors[i, k]
                uk = points[ni, :] - p0
                X[i, 0, k] = vec_dot_3D(uk, e1i)
                X[i, 1, k] = vec_dot_3D(uk, e2i)

    for i in prange(FIRST_POINTS, npts):
        if mask is None or not mask[i]:
            p0 = points[i, :]
            e1i = e1[i, :]
            e2i = e2[i, :]
            for k in range(nhbrs):
                ni = neighbors[i, k]
                uk = points[ni, :] - p0
                X[i, 0, k] = vec_dot_3D(uk, e1i)
                X[i, 1, k] = vec_dot_3D(uk, e2i)

    return X


def lonlat2xyz(Lon, Lat, r, deg2rad=False, return_array=False):
    """
    Convert lon, lat positions to x, y, z.

    Parameters
    ----------
    Lon : np.ndarray, shape=(nx, ny)
        meshgrid of longitude.
    Lat : np.ndarray, shape=(nx, ny)
        meshgrid of latitude.
    r : float
        radius.
    deg2rad : bool, optional
        flag to convert from degree to radians. Lon, Lat must either
        already be in radians, or this flag must be set to True.
        The default is False.
    return_array : bool, optional
        flag to return stacked array instead of tuple. The default is False.

    Returns
    -------
    tuple or np.ndarray
        either tuple or stacked array containing meshgrid of X, Y, Z position.

    """

    if deg2rad:
        Lon = np.deg2rad(Lon)
        Lat = np.deg2rad(Lat)

    Xp = r * np.cos(Lat) * np.cos(Lon)
    Yp = r * np.cos(Lat) * np.sin(Lon)
    Zp = r * np.sin(Lat)

    if return_array:
        return np.stack((Xp, Yp, Zp), axis=-1)
    else:
        return Xp, Yp, Zp


def local_basis_S2(Lon, Lat, deg2rad=False):
    """
    Create a local basis on the surface of the sphere (S2) in x, y, z coords.

    Parameters
    ----------
    Lon : np.ndarray, shape=(nx, ny)
        meshgrid of longitude.
    Lat : np.ndarray, shape=(nx, ny)
        meshgrid of latitude.
    deg2rad : bool, optional
        flag to convert from degree to radians. Lon, Lat must either
        already be in radians, or this flag must be set to True.
        The default is False.

    Returns
    -------
    e1 : np.ndarray, shape=(nx, ny, 2)
        local basis vector in the "east" direction.
    e2 : np.ndarray, shape=(nx, ny, 2)
        local basis vector in the "north" direction.

    """
    nx, ny = Lon.shape
    if deg2rad:
        Lon = np.deg2rad(Lon)
        Lat = np.deg2rad(Lat)

    sinLon = np.sin(Lon)
    sinLat = np.sin(Lat)
    cosLon = np.cos(Lon)
    cosLat = np.cos(Lat)
    e1 = np.stack((-sinLon, cosLon, np.zeros((nx, ny), np.float64)), axis=-1)
    e2 = np.stack((-sinLat * cosLon, -sinLat * sinLon, cosLat), axis=-1)

    return e1, e2


def xyz2lonlat(points, r, return_array=False):
    """
    Compute lon-lat coords from points, which are xyz coords.

    Parameters
    ----------
    points : np.ndarray, shape = (npts, 3)
        array containing points in xyz coords.
    r : float
        radius of sphere.
    return_array : bool, optional
        flag to determine if tuple is returned (False) or array is returned (True).
        The default is False.

    Returns
    -------
    tuple or np.ndarray
        tuple or array containing lon-lat coords of points.

    """
    lon_rad = np.arctan2(points[:, 1], points[:, 0])

    z_ratio = points[:, 2] / r

    z_ratio[z_ratio > 1.0] = 1.0
    z_ratio[z_ratio < -1.0] = -1.0

    lat_rad = np.arcsin(z_ratio)

    if return_array:
        return np.stack((lon_rad, lat_rad), axis=-1)
    else:
        return lon_rad, lat_rad


@njit
def xyz2lonlat_jit(points, r):
    """
    Compute lon-lat coords from points, which are xyz coords. JIT-compiled version.

    Parameters
    ----------
    points : np.ndarray, shape = (npts, 3)
        array containing points in xyz coords.
    r : float
        radius of sphere.

    Returns
    -------
    np.ndarray, shape=(npts, 2)
        tuple or array containing lon-lat coords of points.

    """
    lon = np.arctan2(points[:, 1], points[:, 0])

    z_ratio = points[:, 2] / r

    z_ratio[z_ratio > 1.0] = 1.0
    z_ratio[z_ratio < -1.0] = -1.0

    lat = np.arcsin(z_ratio)

    return np.stack((lon, lat), axis=-1)


def local_basis_icosphere(lons, lats, deg2rad=False):
    """
    Create a local basis on the surface of the icosphere x, y, z coords.

    Parameters
    ----------
    lons : np.ndarray, shape=(npts,)
        collection of longitude points defining the mesh.
    lats : np.ndarray, shape=(npts,)
        collection of latitude points defining the mesh.
    deg2rad : bool, optional
        flag to convert from degree to radians. lons, lats, must either
        already be in radians, or this flag must be set to True.
        The default is False.

    Returns
    -------
    e1 : np.ndarray, shape=(npts, 2)
        local basis vector in the "east" direction.
    e2 : np.ndarray, shape=(npts, 2)
        local basis vector in the "north" direction.

    """
    npts = len(lons)
    if deg2rad:
        lons = np.deg2rad(lons)
        lats = np.deg2rad(lats)

    sinLon = np.sin(lons)
    sinLat = np.sin(lats)
    cosLon = np.cos(lons)
    cosLat = np.cos(lats)
    e1 = np.stack((-sinLon, cosLon, np.zeros((npts,), np.float64)), axis=-1)
    e2 = np.stack((-sinLat * cosLon, -sinLat * sinLon, cosLat), axis=-1)

    return e1, e2


def fill_nans_and_get_mask(arrs, fill_value=0.0):
    """
    For a collection of arrs with the same shape and same indices of
    nan values, obtain a boolean mask corresponding to the nan values
    and set the values of each arr to 'fill_value' wherever that mask
    is true. It is expected the same mask is to be applied for every slice of
    the leading dimension. Designed for data where the leading dimension is
    time, all other dimensions are spatial dimensions, and a spatial mask
    is being applied.

    Parameters
    ----------
    arrs : tuple
        tuple of np.ndarrays, each should have the same size.
    fill_value : float, optional
        value to fill . The default is 0.0.

    Returns
    -------
    tuple
        tuple containing filled arrays and mask. Arrays will be
        shape=(nt, nx, ny), mask shape=(nx, ny).

    """

    arr0 = arrs[0]
    mask = np.isnan(arr0[0])
    arr0[:, mask] = 0.0
    arr_list = [arr0]
    for arr in arrs[1:]:
        arr[:, mask] = 0.0
        arr_list.append(arr)

    return (*arr_list, mask)


@njit(parallel=True)
def interpolate_mask_grid(mask, xp, yp, xq, yq):
    """
    Interpolate mask defined over (xp, yp), to a new mask, defined over (xq, yq).
    Points falling outside of initial grid will be masked.

    Parameters
    ----------
    mask : np.ndarray, shape=(nxp, nyq)
        array of bools defining mask.
    xp : np.ndarray, shape=(nxp,)
        array containing x values at which mask was defined.
    yp : np.ndarray, shape=(nyp,)
        array containing y values at which mask was defined.
    xq : np.ndarray, shape=(nxq,)
        array containing x values at which new mask will defined.
    yq : np.ndarray, shape=(nyq,)
        array containing y values at which new mask will defined.

    Returns
    -------
    new_mask : np.ndarray, shape=(nxq, nyq)
        array of bools defining new mask.

    """
    dx = xp[1] - xp[0]
    dy = yp[1] - yp[0]

    xmin, ymin = xp[0], yp[0]
    xmax, ymax = xp[-1], yp[-1]
    nx, ny = len(xp), len(yp)

    nxq, nyq = len(xq), len(yq)
    new_mask = np.zeros((nxq, nyq), numba.boolean)

    for i in prange(nxq):
        xqi = xq[i]

        # if point is out of bounds, mask
        if xqi > xmax or xqi < xmin:
            mask[i, :] = True
            continue

        # find index of nearest x point from original grid
        ix = round((xqi - xmin) / dx)

        if ix > nx - 1:
            ix = nx - 1

        for j in range(nyq):
            yqj = yq[j]

            # if point is out of bounds, mask
            if yqj > ymax or yqj < ymin:
                mask[i, j] = True
                continue

            # find index of nearest y point from original grid
            iy = round((yqj - ymin) / dy)

            if iy > ny - 1:
                iy = ny - 1
            # nearest interpolation
            new_mask[i, j] = mask[ix, iy]

    return new_mask


@njit(parallel=True)
def binary_mask_dilation(mask, corners=False):
    """
    Performs a binary dilation on a 2D structured grid boolean mask.

    Parameters
    ----------
    mask : np.ndarray, shape=(nx, ny)
        boolean mask.
    corners : bool, optional
        if False (default), uses 4 cardinal neighbors.
        If True, 4 cardinal neighbors plus 4 corner neighbors.

    Returns
    -------
    dilated_masl : np.ndarray, shape=(nx, ny)
        the dilated mask.
    """

    nx, ny = mask.shape
    dilated_mask = mask.copy()

    for i in prange(nx):
        for j in range(ny):
            # check main neighbors

            if mask[i, j]:
                continue

            if i > 0 and mask[i - 1, j]:
                dilated_mask[i, j] = True
                continue

            if i < nx - 1 and mask[i + 1, j]:
                dilated_mask[i, j] = True
                continue

            if j > 0 and mask[i, j - 1]:
                dilated_mask[i, j] = True
                continue

            if j < ny - 1 and mask[i, j + 1]:
                dilated_mask[i, j] = True
                continue

            # check corner neighbors if corners=True
            if corners:
                if i > 0 and j > 0 and mask[i - 1, j - 1]:
                    dilated_mask[i, j] = True
                    continue

                if i > 0 and j < ny - 1 and mask[i - 1, j + 1]:
                    dilated_mask[i, j] = True
                    continue

                if i < nx - 1 and j > 0 and mask[i + 1, j - 1]:
                    dilated_mask[i, j] = True
                    continue

                if i < nx - 1 and j < ny - 1 and mask[i + 1, j + 1]:
                    dilated_mask[i, j] = True

    return dilated_mask


@njit(parallel=True)
def interpolate_mask_mesh(mask, xp, yp, mesh, convert=True, r=6371.0, wrap_x=True):
    """
    Interpolate mask defined over (xp, yp), to a new mask, defined over mesh.
    Points falling outside of initial grid will be masked.

    Parameters
    ----------
    mask : np.ndarray, shape=(nxp, nyq)
        array of bools defining mask.
    xp : np.ndarray, shape=(nxp,)
        array containing x values at which mask was defined.
    yp : np.ndarray, shape=(nyp,)
        array containing y values at which mask was defined.
    mesh : np.ndarray, shape=(npts, 2) or (npts, 3)
        array containing (x, y) points defining the mesh.
    convert : bool, optional
        boolean determining if the points need to be converted from xyz to
        lon lat. The default is True.
    r : float, optional
        radius of sphere if mesh is defined on sphere. The default is 6371.0.
    wrap_x : bool, optional
        flag that determines if data is periodic in x, for data defined on
        the sphere, this should be set to True. The default is True.

    Returns
    -------
    new_mask : np.ndarray, shape=(npts,)
        array of bools defining new mask.

    """
    dx = xp[1] - xp[0]
    dy = yp[1] - yp[0]

    xmin, xmax = xp[0], xp[-1]
    ymin, ymax = yp[0], yp[-1]

    if wrap_x:
        xmax += dx

    nx, ny = len(xp), len(yp)

    npts = len(mesh)
    new_mask = np.zeros(npts, numba.boolean)

    if convert:
        mesh = xyz2lonlat_jit(mesh, r)

    for i in prange(npts):
        xqi, yqi = mesh[i, :]

        # if point is out of bounds, mask
        if xqi > xmax or xqi < xmin or yqi > ymax or yqi < ymin:
            new_mask[i] = True
            continue

        # find index of nearest x point from original grid, deal with boundary
        ix = round((xqi - xmin) / dx)
        if ix > nx - 1:
            ix = nx - 1

        # find index of nearest y point from original grid, deal with boundary
        iy = round((yqi - ymin) / dy)
        if iy > ny - 1:
            iy = ny - 1

        # nearest interpolation
        new_mask[i] = mask[ix, iy]

    return new_mask


@njit(parallel=True)
def binary_mask_dilation_mesh(mask, neighbors, first_points=12, less_neighbors=1):
    """
    Performs a binary dilation on a boolean mask over a mesh. The neighbors
    array contains the connected neighbors for each index of mask. If first_points > 0,
    the first points correspdonding to that value will only check n - less_neighbors
    since this function is designed for an icosphere mesh.

    Parameters
    ----------
    mask : np.ndarray, shape=(npts,)
        boolean mask.
    neighbors : np.ndarray, shape=(npts, n)
        array containing neighbors for each vertex.
    first_points : int, optional
        determines how many points will be checked using n - less_neighbors
        neighbors. For the icosphere, the first 12 points will have 5 neighbors
        while all the rest will have 6. If every point in your mesh has the same
        number of neighbors, set this to 0. Must be nonnegative. The default is 12.
    less_neighbors : int, optional
        how many less neighbors will be checked for the first `first_points`
        points. Will have no affect if first_points=0. The default is 1.

    Returns
    -------
    np.ndarray, shape=(npts,)
        the dilated mask.
    """

    npts, n = neighbors.shape
    dilated_mask = mask.copy()

    if first_points > 0:
        for i in range(first_points):
            if mask[i]:
                continue

            for k in range(n - less_neighbors):
                if mask[neighbors[i, k]]:
                    dilated_mask[i] = True
                    break

    for i in prange(first_points, npts):
        if mask[i]:
            continue

        for k in range(n):
            if mask[neighbors[i, k]]:
                dilated_mask[i] = True
                break

    return dilated_mask


def icosphere_and_displacements(subdivisions, r=6371.0, normalize_once=False, mask_data=None):
    """
    Generate mesh for icosphere, find neighbors of each vertex, and compute
    displacements of each mesh point and its neighbors, projected onto local
    basis on the sphere.

    Parameters
    ----------
    subdivisions : int
        number of subdivisions. For ftle calculations, 7 or 8 is sufficient.
        Any lower will produce under resolved ftle fields, any higher will
        result in long run times.
    r : float, optional
        radius of the sphere, will default to the radius of the earth (in km).
        The default is 6371.0.
    normalize_once : bool, optional
        flag determining how normalization is handled.
        If normalize_once is True, normalizing to the surface of the sphere will
        only happen at the very end, if not, normalization happens at every
        level of subdivision. The former is faster, the latter will produce
        more evenly spaced points. The default is False.
    mask_data : None or tuple, optional
        for masked data, pass in a tuple containing, pass in a boolean mask
        corresponding to nan values (True indicates a nan value) and the lon, lat,
        grid values which that mask was defined onn If mask_data is not None,
        a mesh_mask and dilated version will be returned. Pass these into
        flowmap_ND() and the ftle_icosphere() function respectively.
        The default is None.

    Returns
    -------
    verts : np.ndarray, shape=(10 * 4**subdivisions + 2, 3)
        array containing vertices of mesh.
    neighbors : np.ndarray, shape=(10 * 4**subdivisions + 2, 6)
        array containing indices of neighbors for each mesh point.
    X : np.ndarray, shape=(10 * 4**subdivisions + 2, 2, 6)
        array containing displacements for the neighbors of each mesh point.
    mesh_mask : np.ndarray, shape=(len(verts),), optional
        if mask is not None, a mask interpolated onto the mesh will be returned
        that can be passed into the flowmap_ND() function.
    dilated_mask : np.ndarray, shape=(len(verts),), optional
        if mask is not None, a dilated version of mesh_mask will be returned, which
        can be passed into ftle_icosphere() to avoid erroneous computations at
        mask boundaries.

    """

    # generate icosphere mesh
    verts, faces = icosphere(subdivisions, r, normalize_once=normalize_once)

    # find neighbors
    neighbors = find_neighbors(faces, len(verts))

    # make sure verts is float64 for particle integration
    verts = verts.astype(np.float64)

    # convert mesh to lon lat and compute local basis on the sphere
    mesh_lons_rad, mesh_lats_rad = xyz2lonlat(verts, r, return_array=False)
    e1, e2 = local_basis_icosphere(mesh_lons_rad, mesh_lats_rad, deg2rad=False)

    if mask_data is not None:
        # unpack grid data
        grid_mask, lon, lat = mask_data

        # interpolate the grid mask onto the mesh, will be returned
        mask = interpolate_mask_mesh(grid_mask, lon, lat, verts, r=r)

        # dilate mask to avoid erroneous computations, will be returned
        dilated_mask = binary_mask_dilation_mesh(mask, neighbors)

        # compute displacements of each mesh point and its neighbors
        X = displacements_proj_ico(verts, e1, e2, neighbors, mask=dilated_mask)
        return verts, neighbors, X, mask, dilated_mask
    else:
        # compute displacements of each mesh point and its neighbors
        X = displacements_proj_ico(verts, e1, e2, neighbors)
        return verts, neighbors, X

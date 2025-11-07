import numpy as np
from math import log
from numba import njit, prange, float64, int32
from .utils import (
    composite_simpsons,
    unravel_index,
    finite_diff_ND,
    gradF_stencil_2D,
    gradF_aux_stencil_2D,
    gradF_main_stencil_2D,
    gradUV_stencil_2D,
    eigvalsh_max_2D,
    inv_2D,
    vec_dot_3D,
    lonlat2xyz,
    local_basis_S2,
)


# %% Finite-time
@njit(parallel=True)
def ftle_grid_2D(flowmap, T, dx, dy, mask=None):
    """
    Compute 2D FTLE field from flowmap which is solution of ode over an initial grid defined by x
    and y for integration time T.

    Parameters
    ----------
    flowmap : np.ndarray, shape = (nx, ny, 2)
        array containing final positions of trajectories of initial grid from t0 to t0+T.
    T : float
        integration time.
    dx : float
        grid spacing in x-direction.
    dy : float
        grid spacing in y-direction.
    mask : None or np.ndarray, shape = (nx, ny), optional
        for masked data, pass in a boolean mask corresponding to nan values
        (True indicates a nan value). To avoid erroneous computations at mask
        boundaries, mask passed in should be dilated using the
        binary_mask_dilation function from the utils module. The default is None.

    Returns
    -------
    ftle : np.ndarray, shape = (nx,ny)
        array containing ftle values.

    """

    nx, ny = flowmap.shape[:-1]
    ftle = np.zeros((nx, ny), float64)
    scaling = 1 / (2 * abs(T))
    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            if mask is None or not mask[i, j]:
                dxdx, dxdy, dydx, dydy = gradF_stencil_2D(flowmap, i, j, dx, dy)

                off_diagonal = dxdx * dxdy + dydx * dydy
                C = np.array([[dxdx**2 + dydx**2, off_diagonal], [off_diagonal, dxdy**2 + dydy**2]])

                max_eig = eigvalsh_max_2D(C)
                if max_eig > 1:
                    ftle[i, j] = scaling * log(max_eig)

    return ftle


@njit(parallel=True)
def C_tensor_2D(flowmap_aux, dx, dy, h=1e-5, mask=None):
    """
    Compute eigenvalues and eigenvectors of Cauchy Green tensor in 2D from flowmap_aux which is
    solution of ode over an auxilary grid defined by x,y +-h for integration time T.

    Parameters
    ----------
    flowmap_aux : np.ndarray, shape = (nx, ny, n_aux, 2)
        array containing final positions of initial grid with aux grid spacing h from t0 to t0+T.
    dx : float
        grid spacing in x-direction.
    dy : float
        grid spacing in y-direction.
    h : float, optional
        aux grid spacing. The default is 1e-5.
    mask : None or np.ndarray, shape = (nx, ny), optional
        for masked data, pass in a boolean mask corresponding to nan values
        (True indicates a nan value). The default is None.

    Returns
    -------
    C : np.ndarray, shape = (nx, ny, 3)
        array containing C11, C12, C22 components of Cauchy Green tensor.

    """

    nx, ny = flowmap_aux.shape[:2]
    C = np.zeros((nx, ny, 3), float64)
    for i in prange(2, nx - 2):
        for j in range(2, ny - 2):
            if mask is None or not mask[i, j]:
                dxdx_aux, dxdy_aux, dydx_aux, dydy_aux = gradF_aux_stencil_2D(flowmap_aux, i, j, h)

                C[i, j, :] = np.array(
                    [
                        dxdx_aux**2 + dydx_aux**2,
                        dxdx_aux * dxdy_aux + dydx_aux * dydy_aux,
                        dxdy_aux**2 + dydy_aux**2,
                    ]
                )

    return C


@njit(parallel=True)
def C_eig_aux_2D(flowmap_aux, dx, dy, h=1e-5, eig_main=True, mask=None):
    """
    Compute eigenvalues and eigenvectors of Cauchy Green tensor in 2D from flowmap_aux which is
    solution of ode over an auxilary grid defined by x,y +-h for integration time T.

    Parameters
    ----------
    flowmap_aux : np.ndarray, shape = (nx,ny,n_aux,2)
        array containing final positions of initial grid with aux grid spacing h from t0 to t0+T.
    dx : float
        grid spacing in x-direction.
    dy : float
        grid spacing in y-direction.
    h : float, optional
        aux grid spacing. The default is 1e-5.
    eig_main : boolean, optional
        flag to determine if eigevalues are computed from main grid. The default is True.
    mask : None or np.ndarray, shape = (nx, ny), optional
        for masked data, pass in a boolean mask corresponding to nan values
        (True indicates a nan value). To avoid erroneous computations at mask
        boundaries (if eig_main=True), mask passed in should be dilated using the
        binary_mask_dilation function from the utils module. The default is None.

    Returns
    -------
    eigvals : np.ndarray, shape = (nx,ny,2)
        array containing eigenvalues values.
    eigvecs : np.ndarray, shape = (nx,ny,2,2)
        array containing eigenvectors.

    """

    nx, ny = flowmap_aux.shape[:-2]
    eigvals = np.zeros((nx, ny, 2), float64)
    eigvecs = np.zeros((nx, ny, 2, 2), float64)
    if eig_main:
        for i in prange(2, nx - 2):
            for j in range(2, ny - 2):
                if mask is None or not mask[i, j]:
                    dxdx_aux, dxdy_aux, dydx_aux, dydy_aux = gradF_aux_stencil_2D(
                        flowmap_aux, i, j, h
                    )

                    dxdx_main, dxdy_main, dydx_main, dydy_main = gradF_main_stencil_2D(
                        flowmap_aux, i, j, dx, dy
                    )

                    off_diagonal_aux = dxdx_aux * dxdy_aux + dydx_aux * dydy_aux
                    C_aux = np.array(
                        [
                            [dxdx_aux**2 + dydx_aux**2, off_diagonal_aux],
                            [off_diagonal_aux, dxdy_aux**2 + dydy_aux**2],
                        ]
                    )

                    off_diagonal_main = dxdx_main * dxdy_main + dydx_main * dydy_main
                    C_main = np.array(
                        [
                            [dxdx_main**2 + dydx_main**2, off_diagonal_main],
                            [off_diagonal_main, dxdy_main**2 + dydy_main**2],
                        ]
                    )

                    _, evecs_tmp = np.linalg.eigh(C_aux)
                    evals_tmp = np.linalg.eigvalsh(C_main)
                    eigvals[i, j, :] = evals_tmp
                    eigvecs[i, j, :, :] = evecs_tmp

    else:
        for i in prange(1, nx - 1):
            for j in range(1, ny - 1):
                if mask is None or not mask[i, j]:
                    dxdx, dxdy, dydx, dydy = gradF_aux_stencil_2D(flowmap_aux, i, j, h)

                    off_diagonal = dxdx * dxdy + dydx * dydy
                    C = np.array(
                        [[dxdx**2 + dydx**2, off_diagonal], [off_diagonal, dxdy**2 + dydy**2]]
                    )

                    evals_tmp, evecs_tmp = np.linalg.eigh(C)
                    eigvals[i, j, :] = evals_tmp
                    eigvecs[i, j, :, :] = evecs_tmp

    return eigvals, eigvecs


@njit(parallel=True)
def C_eig_2D(flowmap, dx, dy, mask=None):
    """
    Compute eigenvalues and eigenvectors of Cauchy Green tensor in 2D from flowmap which is
    solution of ode over a grid defined by x,y for integration time T.

    Parameters
    ----------
    flowmap : np.ndarray, shape = (nx, ny, 2)
        array containing final positions of initial grid from t0 to t0+T.
    dx : float
        grid spacing in x-direction.
    dy : float
        grid spacing in y-direction.
    mask : None or np.ndarray, shape = (nx, ny), optional
        for masked data, pass in a boolean mask corresponding to nan values
        (True indicates a nan value). To avoid erroneous computations at mask
        boundaries, mask passed in should be dilated using the
        binary_mask_dilation function from the utils module. The default is None.

    Returns
    -------
    eigvals : np.ndarray, shape = (nx, ny, 2)
        array containing eigenvalues.
    eigvecs : np.ndarray, shape = (nx, ny, 2, 2)
        array containing eigenvectors.

    """

    nx, ny = flowmap.shape[:-1]
    eigvals = np.zeros((nx, ny, 2), float64)
    eigvecs = np.zeros((nx, ny, 2, 2), float64)
    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            if mask is None or not mask[i, j]:
                dxdx, dxdy, dydx, dydy = gradF_stencil_2D(flowmap, i, j, dx, dy)

                off_diagonal = dxdx * dxdy + dydx * dydy
                C = np.array([[dxdx**2 + dydx**2, off_diagonal], [off_diagonal, dxdy**2 + dydy**2]])

                evals_tmp, evecs_tmp = np.linalg.eigh(C)
                eigvals[i, j, :] = evals_tmp
                eigvecs[i, j, :, :] = evecs_tmp

    return eigvals, eigvecs


def ftle_from_eig(eigval_max, T):
    """
    Compute FTLE field from eigval_max where eigval_max is eigenvalue of Cauchy-Green tensor
    computed using integration time T.

    Parameters
    ----------
    eigval_max : np.ndarray, shape = (nx,ny)
        maximum eigenvalue of Cauchy-Green tensor computed using integration time T.
    T : float
        integration time.

    Returns
    -------
    ftle : np.ndarray, shape = (nx,ny)
        array containing ftle values.

    """

    ftle = np.zeros(eigval_max.shape)
    ftle[np.nonzero(eigval_max > 1)] = np.log(eigval_max[np.nonzero(eigval_max > 1)]) / (2 * abs(T))

    return ftle


@njit(parallel=True)
def lavd_grid_2D(
    flowmap_n,
    tspan,
    T,
    vort_interp,
    xrav,
    yrav,
    period_x=0.0,
    period_y=0.0,
    mask=None,
):
    """
    Compute LAVD from flowmap_n where flowmap_n contains trajectories computed over gridpoints
    defined by xrav,yrav for an integration time T and trajectories are returned at times given by
    tspan. vort_interp is an interpolant function of vorticity over (at least) that time window.

    Parameters
    ----------
    flowmap_n : np.ndarray, shape = (nx, ny, n, 2)
        array containing trajectories of initial grid from t0 to t0+T.
    tspan : np.ndarray, shape = (n,)
        array containing n times corresponding to axis=2 of flowmap_n.
    T : float
        integration time.
    vort_interp : jit-callable
        interpolant function of vorticity which must be (at least) defined over all values of
        flowmap_n and times from tspan.
    xrav : np.ndarray, shape = (nx*ny,))
        array containing raveled or flattened meshgrid X, can be obtained using X.ravel()
        or X.flatten().
    yrav : np.ndarray, shape = (nx*ny,))
        array containing raveled or flattened meshgrid Y, can be obtained using Y.ravel()
        or Y.flatten().
    period_x : float
        value for period in x-direction, if not periodic, set equal to 0.0. The default is 0.0.
    period_y : float
        value for period in y-direction, if not periodic, set equal to 0.0. The default is 0.0.
    mask : None or np.ndarray, shape = (nx, ny), optional
        for masked data, pass in a boolean mask corresponding to nan values
        (True indicates a nan value). The default is None.

    Returns
    -------
    lavd : np.ndarray, shape = (nx,ny)
        array containing lavd values.

    """

    nx, ny, n = flowmap_n.shape[:-1]
    npts = len(xrav)
    vort_avg = np.zeros(n, float64)
    for k in prange(n):
        gpts = np.zeros((npts, 3), float64)
        gpts[:, 0] = tspan[k]
        gpts[:, 1] = xrav
        gpts[:, 2] = yrav

        vort_k = vort_interp(gpts)
        vort_avg[k] = np.mean(vort_k)

    lavd = np.zeros((nx, ny), float64)
    dt = abs(tspan[1] - tspan[0])
    if period_x + period_y == 0.0:
        for i in prange(nx):
            for j in range(ny):
                if mask is None or not mask[i, j]:
                    pts = np.zeros((n, 3), float64)
                    pts[:, 0] = tspan
                    pts[:, 1:] = flowmap_n[i, j, :, :]
                    vort_traj = vort_interp(pts)
                    integrand = np.abs(vort_traj - vort_avg)
                    lavd[i, j] = composite_simpsons(integrand, dt)
    elif period_x and period_y:
        for i in prange(nx):
            for j in range(ny):
                if mask is None or not mask[i, j]:
                    pts = np.zeros((n, 3), float64)
                    pts[:, 0] = tspan
                    pts[:, 1] = flowmap_n[i, j, :, 0] % period_x
                    pts[:, 2] = flowmap_n[i, j, :, 1] % period_y
                    vort_traj = vort_interp(pts)
                    integrand = np.abs(vort_traj - vort_avg)
                    lavd[i, j] = composite_simpsons(integrand, dt)
    elif period_x:
        for i in prange(nx):
            for j in range(ny):
                if mask is None or not mask[i, j]:
                    pts = np.zeros((n, 3), float64)
                    pts[:, 0] = tspan
                    pts[:, 1] = flowmap_n[i, j, :, 0] % period_x
                    pts[:, 2] = flowmap_n[i, j, :, 1]
                    vort_traj = vort_interp(pts)
                    integrand = np.abs(vort_traj - vort_avg)
                    lavd[i, j] = composite_simpsons(integrand, dt)
    elif period_y:
        for i in prange(nx):
            for j in range(ny):
                if mask is None or not mask[i, j]:
                    pts = np.zeros((n, 3), float64)
                    pts[:, 0] = tspan
                    pts[:, 1] = flowmap_n[i, j, :, 0]
                    pts[:, 2] = flowmap_n[i, j, :, 1] % period_y
                    vort_traj = vort_interp(pts)
                    integrand = np.abs(vort_traj - vort_avg)
                    lavd[i, j] = composite_simpsons(integrand, dt)

    return lavd


@njit(parallel=True)
def ftle_grid_ND(flowmap, IC, T, dX):
    """
    Compute ND FTLE field from flowmap which is solution of ode over an initial grid defined by IC
    for integration time T.

    Parameters
    ----------
    flowmap : np.ndarray, shape = (nx_1*nx_2*...*nx_ndims,ndims)
        array containing final positions of initial grid from t0 to t0+T.
    IC : np.ndarray, shape = (nx_1,nx_2,...,nx_ndims,ndims)
        initial condition array.
    T : float
        integration time.
    dX : np.ndarray, shape = (ndims,)
        array containing spacing in each x_i direction for i = 1,...,ndims.

    Returns
    -------
    ftle : np.ndarray, shape = (nx_1*nx_2*...*nx_ndims,)
        array containing ftle values.

    """

    grid_shape = np.array(IC.shape[:-1])
    ndims = IC.shape[-1]
    npts = np.prod(grid_shape)
    ftle = np.zeros(npts)
    denom = 2 * abs(T)
    for k in prange(npts):
        inds = unravel_index(k, grid_shape)
        dXdir = np.zeros(ndims, int32)
        for ii, ind in enumerate(inds):
            if ind == 0:
                dXdir[ii] = 1
            elif ind == grid_shape[ii] - 1:
                dXdir[ii] = -1
            else:
                dXdir[ii] = 0
        Df = np.zeros((ndims, ndims), float64)
        for i in range(ndims):
            for j in range(ndims):
                Df[i, j] = finite_diff_ND(flowmap[:, i], inds, dX[j], j, grid_shape, dXdir[j])

        C = np.dot(Df.T, Df)
        max_eig = np.linalg.eigvalsh(C)[-1]
        if max_eig > 1:
            ftle[k] = log(max_eig) / denom

    return ftle


# %% least-squares ftle
@njit(parallel=True)
def displacement_array_proj(points, e1, e2, full=True, mask=None):
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
    full : bool, optional
        flag to determine if full 8 (True) neighbors of grid point are used or
        just 4 (False). The default is True.
    mask : None or np.ndarray, shape = (nx, ny), optional
        for masked data, pass in a boolean mask corresponding to nan values
        (True indicates a nan value). To avoid erroneous computations at mask
        boundaries, mask passed in should be dilated using the
        binary_mask_dilation function from the utils module. The default is None.

    Returns
    -------
    X : np.ndarray, shap=(nx, ny, 2, n)
        array corresponding to displacements of n neighbors for each point.

    """

    nx, ny = points.shape[:-1]
    if full:
        n = 8
        stencil = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1]])
    else:
        n = 4
        stencil = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    X = np.zeros((nx, ny, 2, n), float64)

    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            if mask is None or not mask[i, j]:
                p0 = points[i, j, :]
                e1i = e1[i, j, :]
                e2i = e2[i, j, :]
                for k in range(n):
                    ii, jj = stencil[k]
                    uk = points[i + ii, j + jj, :] - p0
                    X[i, j, 0, k] = vec_dot_3D(uk, e1i)
                    X[i, j, 1, k] = vec_dot_3D(uk, e2i)
    return X


@njit(parallel=True)
def _displacement_array(points, full=True, mask=None):
    """Compute displacement array."""

    nx, ny = points.shape[:-1]
    if full:
        n = 8
        stencil = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1]])
    else:
        n = 4
        stencil = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    X = np.zeros((nx, ny, 3, n), float64)

    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            if mask is None or not mask[i, j]:
                p0 = points[i, j, :]
                for k in range(n):
                    ii, jj = stencil[k, :]
                    X[i, j, :, k] = points[i + ii, j + jj, :] - p0
    return X


@njit(parallel=True)
def _ftle_lsq_opt_2D(X, Ytilde, T, mask=None):
    """
    Compute FTLE on a 2D manifold using the least-squares estimate. The X
    array contains the initial displacements calculated in the embedded space,
    projected onto a local basis of the tangent space on the manifold. The
    Y array contains the final displacements calculated in the embedded space.
    This function performs all linear algebra manually to optimize memory
    access and avoid cache misses. For more info on the background theory,
    see Lekien and Ross. "The computation of finite-time Lyapunov exponents on
    unstructured meshes and for non-Euclidean manifolds". doi: 10.1063/1.3278516.

    Parameters
    ----------
    X : np.ndarray, shape = (nx, ny, 2, n)
        initial displacements of n nearby points.
    Ytilde : np.ndarray, shape = (nx, ny, 3, n)
        final displacements of n nearby points.
    T : float
        integration time.

    Returns
    -------
    ftle : np.ndarray, shape = (nx, ny)
        ftle values.

    """

    nx, ny = X.shape[:2]
    n = X.shape[-1]
    ftle = np.zeros((nx, ny), float64)
    scaling = 1 / (2 * abs(T))

    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            # The operations here are really just the following:
            # Mtilde = (Ytilde @ X.T) @ (np.linalg.inv(X @ X.T))
            # C = Mtilde.T @ Mtilde
            # max_eig = np.linalg.eigvalsh(C)[-1]
            # ftle[i,j] = 1 / (2 * |T|) * log(max_eig)
            #
            # We perform the matrix multiplication manually to avoid memory
            # access problems and significantly speed up the computations
            if mask is None or not mask[i, j]:
                # XXT = X @ X.T
                XXT = np.zeros((2, 2), float64)
                for r in range(2):
                    for c in range(2):
                        val = 0.0
                        for k in range(n):
                            val += X[i, j, r, k] * X[i, j, c, k]
                        XXT[r, c] = val

                XXT_inv = inv_2D(XXT)

                # YXT = Ytilde @ X.T
                YXT = np.zeros((3, 2), float64)
                for r in range(3):
                    for c in range(2):
                        val = 0.0
                        for k in range(n):
                            val += Ytilde[i, j, r, k] * X[i, j, c, k]
                        YXT[r, c] = val

                # Mtilde = YXT @ XXT_inv
                Mtilde = np.zeros((3, 2), float64)
                for r in range(3):
                    for c in range(2):
                        val = 0.0
                        for k in range(2):
                            val += YXT[r, k] * XXT_inv[k, c]
                        Mtilde[r, c] = val

                # C = Mtilde.T @ Mtilde
                C = np.zeros((2, 2), float64)
                for r in range(2):
                    for c in range(2):
                        val = 0.0
                        for k in range(3):
                            val += Mtilde[k, r] * Mtilde[k, c]
                        C[r, c] = val

                max_eig = eigvalsh_max_2D(C)

                if max_eig > 1:
                    ftle[i, j] = scaling * log(max_eig)

    return ftle


@njit
def ftle_grid_S2(
    Lon,
    Lat,
    flowmap,
    T,
    r=6371.0,
    full=True,
    X=None,
    deg2rad=True,
    mask=None,
):
    """
    Computes a more accurate approximation of FTLE on the surface of the
    sphere (S2). Uses the least-squares approximation of the FTLE based
    on grid cell displacements in 3D space. For more info on the background
    theory, see Lekien and Ross, "The computation of finite-time Lyapunov
    exponents on unstructured meshes and for non-Euclidean manifolds".
    doi: 10.1063/1.3278516.

    Parameters
    ----------
    Lon : np.ndarray, shape=(nx, ny)
        meshgrid of longitude.
    Lat : np.ndarray, shape=(nx, ny)
        meshgrid of latitude.
    flowmap : np.ndarray, shape=(nx, ny, 2)
        final particle positions for whole grid in lon-lat coords.
    T : float
        integration time.
    r : float, optional
        radius of the sphere, will default to the radius of the earth (in km).
        The default is 6371.0.
    full : bool, optional
        flag to determine if full 8 neighbors of grid point are used or just 4.
        The default is True.
    X : None or np.ndarray, shape=(nx, ny, 2, n)
        initial displacement array that contains the displacement of each
        grid points from its n neighbors (determined by full). This can be computed
        by first computing
        initial_points = numbacs.utils.lonlat2xy(Lon, Lat, r, deg2rad=True, return_array=True).
        Then, e1, e2 = numbacs.utils.local_basis_S2(Lon, Lat, deg2rad=True).
        Then passing the results to
        X = displacement_array_proj(initial_points, e1, e2, full=full, mask=mask, dilate_mask=dilate_mask).
        This is useful if you are computing ftle in a time series, by passing
        is this array you avoid this redundent computation for every iterate.
        If None, this will be computed internally everytime this function is
        called. The default is None.
    deg2rad : bool, optional
        flag to convert from degree to radians. Lon, Lat must either
        already be in radians, or this flag must be set to True. Not relevant
        and will be ignored if X is not None. The default is True.
    mask : None or np.ndarray, shape = (nx, ny), optional
        for masked data, pass in a boolean mask corresponding to nan values
        (True indicates a nan value). To avoid erroneous computations at mask
        boundaries, mask passed in should be dilated using the
        binary_mask_dilation function from the utils module. The default is None.

    Returns
    -------
    np.ndarray, shape=(nx, ny)
        least-squares approximation of ftle on the sphere.

    """

    advected_points = lonlat2xyz(
        ((flowmap[..., 0] - 180) % 360) - 180, flowmap[..., 1], r, deg2rad=True, return_array=True
    )
    if X is None:
        if deg2rad:
            Lon, Lat = np.deg2rad(Lon), np.deg2rad(Lat)
        initial_points = lonlat2xyz(Lon, Lat, r, return_array=True)
        e1, e2 = local_basis_S2(Lon, Lat)
        X = displacement_array_proj(initial_points, e1, e2, full=full, mask=mask)

    Ytilde = _displacement_array(advected_points, full=full, mask=mask)
    return _ftle_lsq_opt_2D(X, Ytilde, mask=mask)


# icosphere versions
@njit(parallel=True)
def _displacements_ico(points, neighbors, mask=None):
    """Compute displacement array."""

    FIRST_POINTS = 12
    npts = len(points)
    N = 6
    X = np.zeros((npts, 3, N), float64)

    for i in range(FIRST_POINTS):
        p0 = points[i, :]
        for k in range(N - 1):
            ni = neighbors[i, k]
            X[i, :, k] = points[ni, :] - p0

    for i in prange(FIRST_POINTS, npts):
        p0 = points[i, :]
        for k in range(N):
            ni = neighbors[i, k]
            X[i, :, k] = points[ni, :] - p0

    return X


@njit(parallel=True)
def _ftle_lsq_opt_ico(X, Ytilde, T, mask=None):
    """
    Compute FTLE on a 2D manifold using the least-squares estimate. The X
    array contains the initial displacements calculated in the embedded space,
    projected onto a local basis of the tangent space on the manifold. The
    Y array contains the final displacements calculated in the embedded space.
    This function performs all linear algebra manually to optimize memory
    access and avoid cache misses. For more info on the background theory,
    see Lekien and Ross. "The computation of finite-time Lyapunov exponents on
    unstructured meshes and for non-Euclidean manifolds". doi: 10.1063/1.3278516.

    Parameters
    ----------
    X : np.ndarray, shape = (npts, 2, 6)
        initial displacements of neighbors.
    Ytilde : np.ndarray, shape = (npts, 3, 6)
        final displacements of neighbors.
    T : float
        integration time.

    Returns
    -------
    ftle : np.ndarray, shape = (npts,)
        ftle values.

    """
    FIRST_POINTS = 12
    npts = len(X)
    nhbrs = 6
    scaling = 1 / (2 * abs(T))
    ftle = np.zeros(npts, float64)

    for i in range(FIRST_POINTS):
        if mask is None or not mask[i]:
            # The operations here are really just the following:
            # Mtilde = (Ytilde @ X.T) @ (np.linalg.inv(X @ X.T))
            # C = Mtilde.T @ Mtilde
            # max_eig = np.linalg.eigvalsh(C)[-1]
            # ftle[i,j] = 1 / (2 * |T|) * log(max_eig)
            #
            # We perform the matrix multiplication manually to avoid memory
            # access problems and significantly speed up the computations

            # XXT = X @ X.T
            XXT = np.zeros((2, 2), float64)
            for r in range(2):
                for c in range(2):
                    val = 0.0
                    for k in range(nhbrs - 1):
                        val += X[i, r, k] * X[i, c, k]
                    XXT[r, c] = val

            XXT_inv = inv_2D(XXT)

            # YXT = Ytilde @ X.T
            YXT = np.zeros((3, 2), float64)
            for r in range(3):
                for c in range(2):
                    val = 0.0
                    for k in range(nhbrs - 1):
                        val += Ytilde[i, r, k] * X[i, c, k]
                    YXT[r, c] = val

            # Mtilde = YXT @ XXT_inv
            Mtilde = np.zeros((3, 2), float64)
            for r in range(3):
                for c in range(2):
                    val = 0.0
                    for k in range(2):
                        val += YXT[r, k] * XXT_inv[k, c]
                    Mtilde[r, c] = val

            # C = Mtilde.T @ Mtilde
            C = np.zeros((2, 2), float64)
            for r in range(2):
                for c in range(2):
                    val = 0.0
                    for k in range(3):
                        val += Mtilde[k, r] * Mtilde[k, c]
                    C[r, c] = val

            max_eig = eigvalsh_max_2D(C)

            if max_eig > 1:
                ftle[i] = scaling * log(max_eig)

    for i in prange(FIRST_POINTS, npts):
        if mask is None or not mask[i]:
            # Same operations as above

            # XXT = X @ X.T
            XXT = np.zeros((2, 2), float64)
            for r in range(2):
                for c in range(2):
                    val = 0.0
                    for k in range(nhbrs):
                        val += X[i, r, k] * X[i, c, k]
                    XXT[r, c] = val

            XXT_inv = inv_2D(XXT)

            # YXT = Ytilde @ X.T
            YXT = np.zeros((3, 2), float64)
            for r in range(3):
                for c in range(2):
                    val = 0.0
                    for k in range(nhbrs):
                        val += Ytilde[i, r, k] * X[i, c, k]
                    YXT[r, c] = val

            # Mtilde = YXT @ XXT_inv
            Mtilde = np.zeros((3, 2), float64)
            for r in range(3):
                for c in range(2):
                    val = 0.0
                    for k in range(2):
                        val += YXT[r, k] * XXT_inv[k, c]
                    Mtilde[r, c] = val

            # C = Mtilde.T @ Mtilde
            C = np.zeros((2, 2), float64)
            for r in range(2):
                for c in range(2):
                    val = 0.0
                    for k in range(3):
                        val += Mtilde[k, r] * Mtilde[k, c]
                    C[r, c] = val

            max_eig = eigvalsh_max_2D(C)

            if max_eig > 1:
                ftle[i] = scaling * log(max_eig)

    return ftle


@njit
def ftle_icosphere(flowmap, neighbors, X, T, mask=None):
    """
    Computes FTLE on the surface of the sphere (S2), where the sphere is defined
    by mesh_points which are vertices of an icosphere. Uses the least-squares
    approximation of the FTLE based on mesh cell displacements in 3D space.
    For more info on the background theory, see Lekien and Ross,
    "The computation of finite-time Lyapunov exponents on unstructured meshes
    and for non-Euclidean manifolds". doi: 10.1063/1.3278516.

    Parameters
    ----------
    flowmap : np.ndarray, shape=(npts, 3)
        final particle positions for whole mesh in xyz-coords.
    neighbors : np.ndarray, shape=(npts, 6)
        array containing neighbors of initial mesh points. Can be obtained from the following function -
        mesh_points, neighbors, X = numbacs.utils.icosphere_and_displacements(subdivisions, r=r, mask=mask).
    X : np.ndarray, shape=(npts, 2, 6)
        array containing intial projected displacements.Can be obtained from same function as neighbors -
        mesh_points, neighbors, X = numbacs.utils.icosphere_and_displacements(subdivisions, r=r, mask=mask).
    T : float
        integration time.
    mask : None or np.ndarray, shape = (npts,), optional
        for masked data, pass in a boolean mask corresponding to nan values
        (True indicates a nan value). To avoid erroneous computations at mask
        boundaries, mask passed in should be dilated using the
        binary_mask_dilation_mesh function from the utils module. The default is None.

    Returns
    -------
    np.ndarray, shape=(npts,)
        least-squares approximation of ftle on the sphere.

    """
    Ytilde = _displacements_ico(flowmap, neighbors, mask=mask)
    return _ftle_lsq_opt_ico(X, Ytilde, T, mask=mask)


# %% Instantaneous
@njit(parallel=True)
def ile_2D_func(vel, x, y, t0=None, h=1e-3, mask=None):
    """
    Compute the iLE field from the flow defined by vel which is a jit-callable function, step size
    of h is used in finite differencing.

    Parameters
    ----------
    vel : jit-callable
        function returing interpolated of function value of velocity in x,y directions.
    x : np.ndarray, shape = (nx,)
        array containing x-values.
    y : np.ndarray, shape = (ny,)
        array containing y-values.
    t0: float or None, optional
        time value at which to evaluate if v1,v2 interpolants depend on time, if they do not
        depend on time, set to None. The default is None.
    h : float, optional
        step size to be used in finite differencing. The default is 1e-3.
    mask : None or np.ndarray, shape = (nx, ny), optional
        for masked data, pass in a boolean mask corresponding to nan values
        (True indicates a nan value). The default is None.

    Returns
    -------
    ile : np.ndarray, shape = (nx,ny)
        array containing ile values.

    """

    nx, ny = len(x), len(y)
    ile = np.zeros((nx, ny), float64)
    if t0 is None:
        dx_vec = np.array([h, 0.0], float64)
        dy_vec = np.array([0.0, h], float64)
        for i in prange(1, nx - 1):
            for j in range(1, ny - 1):
                if mask is None or not mask[i, j]:
                    pt = np.array([x[i], y[j]], float64)

                    dudx, dvdx = (vel(pt + dx_vec) - vel(pt - dx_vec)) / (2 * h)
                    dudy, dvdy = (vel(pt + dy_vec) - vel(pt - dy_vec)) / (2 * h)

                    grad_vel = np.array([[dudx, dudy], [dvdx, dvdy]])
                    S = 0.5 * (grad_vel + grad_vel.T)
                    ile[i, j] = eigvalsh_max_2D(S)
    else:
        dx_vec = np.array([0.0, h, 0.0], float64)
        dy_vec = np.array([0.0, 0.0, h], float64)
        for i in prange(1, nx - 1):
            for j in range(1, ny - 1):
                if mask is None or not mask[i, j]:
                    pt = np.array([t0, x[i], y[j]], float64)

                    dudx, dvdx = (vel(pt + dx_vec) - vel(pt - dx_vec)) / (2 * h)
                    dudy, dvdy = (vel(pt + dy_vec) - vel(pt - dy_vec)) / (2 * h)

                    grad_vel = np.array([[dudx, dudy], [dvdx, dvdy]])
                    S = 0.5 * (grad_vel + grad_vel.T)
                    ile[i, j] = eigvalsh_max_2D(S)

    return ile


@njit(parallel=True)
def S_eig_2D_func(vel, x, y, t0=None, h=1e-3, mask=None):
    """
    Compute eigenvalues and eigenvectors of Eulerian rate of strain tensor in 2D from vel which
    is a jit-callable function, step size of h is used for finite differencing.

    Parameters
    ----------
    vel : jit-callable
        function returing interpolated of function value of velocity in x,y directions.
    x : np.ndarray, shape = (nx,)
        array containing x-values.
    y : np.ndarray, shape = (ny,)
        array containing y-values.
    t0: float or None, optional
        time value at which to evaluate if v1,v2 interpolants depend on time, if they do not
        depend on time, set to None. The default is None.
    h : float, optional
        step size to be used in finite differencing. The default is 1e-3.
    mask : None or np.ndarray, shape = (nx, ny), optional
        for masked data, pass in a boolean mask corresponding to nan values
        (True indicates a nan value). The default is None.

    Returns
    -------
    eigvals : np.ndarray, shape = (nx, ny)
        array containg eigenvalues of S.
    eigvecs : np.ndarray, shape = (nx, ny)
        array containing eigenvectors of S.

    """

    nx, ny = len(x), len(y)
    eigvals = np.zeros((nx, ny, 2), float64)
    eigvecs = np.zeros((nx, ny, 2, 2), float64)
    if t0 is None:
        dx_vec = np.array([h, 0.0], float64)
        dy_vec = np.array([0.0, h], float64)
        for i in prange(1, nx - 1):
            for j in range(1, ny - 1):
                if mask is None or not mask[i, j]:
                    pt = np.array([x[i], y[j]], float64)

                    dudx, dvdx = (vel(pt + dx_vec) - vel(pt - dx_vec)) / (2 * h)
                    dudy, dvdy = (vel(pt + dy_vec) - vel(pt - dy_vec)) / (2 * h)
                    grad_vel = np.array([[dudx, dudy], [dvdx, dvdy]])
                    S = 0.5 * (grad_vel + grad_vel.T)
                    evals_tmp, evecs_tmp = np.linalg.eigh(S)
                    eigvals[i, j, :] = evals_tmp
                    eigvecs[i, j, :, :] = evecs_tmp
    else:
        dx_vec = np.array([0.0, h, 0.0], float64)
        dy_vec = np.array([0.0, 0.0, h], float64)
        for i in prange(1, nx - 1):
            for j in range(1, ny - 1):
                if mask is None or not mask[i, j]:
                    pt = np.array([t0, x[i], y[j]], float64)

                    dudx, dvdx = (vel(pt + dx_vec) - vel(pt - dx_vec)) / (2 * h)
                    dudy, dvdy = (vel(pt + dy_vec) - vel(pt - dy_vec)) / (2 * h)
                    grad_vel = np.array([[dudx, dudy], [dvdx, dvdy]])
                    S = 0.5 * (grad_vel + grad_vel.T)
                    evals_tmp, evecs_tmp = np.linalg.eigh(S)
                    eigvals[i, j, :] = evals_tmp
                    eigvecs[i, j, :, :] = evecs_tmp

    return eigvals, eigvecs


@njit(parallel=True)
def S_2D_func(vel, x, y, t0=None, h=1e-3, mask=None):
    """
    Compute Eulerian rate of strain tensor in 2D from vel which is a jit-callable functions,
    step size of h is used for finite differencing.

    Parameters
    ----------
    v1 : jit-callable
        function returing interpolated value of velocity in x-direction.
    v2 : jit-callable
        function returing interpolated value of velocity in y-direction.
    x : np.ndarray, shape = (nx,)
        array containing x-values.
    y : np.ndarray, shape = (ny,)
        array containing y-values.
    t0: float or None, optional
        time value at which to evaluate if v1,v2 interpolants depend on time, if they do not
        depend on time, set to None. The default is None.
    h : float, optional
        step size to be used in finite differencing. The default is 1e-3.
    mask : None or np.ndarray, shape = (nx, ny), optional
        for masked data, pass in a boolean mask corresponding to nan values
        (True indicates a nan value). The default is None.

    Returns
    -------
    S : np.ndarray, shape = (nx,ny)
        S11,S12, and S22 components of S tensor.

    """

    nx, ny = len(x), len(y)
    S = np.zeros((nx, ny, 3), float64)
    if t0 is None:
        dx_vec = np.array([h, 0.0], float64)
        dy_vec = np.array([0.0, h], float64)
        for i in prange(nx):
            for j in range(ny):
                if mask is None or not mask[i, j]:
                    pt = np.array([x[i], y[j]], float64)

                    dudx, dvdx = (vel(pt + dx_vec) - vel(pt - dx_vec)) / (2 * h)
                    dudy, dvdy = (vel(pt + dy_vec) - vel(pt - dy_vec)) / (2 * h)
                    S[i, j, :] = np.array([dudx, 0.5 * (dudy + dvdx), dvdy])
    else:
        dx_vec = np.array([0.0, h, 0.0], float64)
        dy_vec = np.array([0.0, 0.0, h], float64)
        for i in prange(nx):
            for j in range(ny):
                if mask is None or not mask[i, j]:
                    pt = np.array([t0, x[i], y[j]], float64)

                    dudx, dvdx = (vel(pt + dx_vec) - vel(pt - dx_vec)) / (2 * h)
                    dudy, dvdy = (vel(pt + dy_vec) - vel(pt - dy_vec)) / (2 * h)
                    S[i, j, :] = np.array([dudx, 0.5 * (dudy + dvdx), dvdy])

    return S


@njit(parallel=True)
def ile_2D_data(u, v, dx, dy, mask=None):
    """
    Compute the iLE field from the flow defined by u,v over an intial grid defined by x,y.

    Parameters
    ----------
    u : np.ndarray, shape = (nx,ny)
        array containing velocity values in x-direction.
    v : np.ndarray, shape = (nx,ny)
        array containing velocity values in y-direction.
    dx : float
        grid spacing in x-direction.
    dy : float
        grid spacing in y-direction.
    mask : None or np.ndarray, shape = (nx, ny), optional
        for masked data, pass in a boolean mask corresponding to nan values
        (True indicates a nan value). To avoid erroneous computations at mask
        boundaries, mask passed in should be dilated using the
        binary_mask_dilation function from the utils module. The default is None.

    Returns
    -------
    ile : np.ndarray, shape = (nx,ny)
        array containing ile values.

    """

    nx, ny = u.shape
    ile = np.zeros((nx, ny))
    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            if mask is None or not mask[i, j]:
                dudx, dudy, dvdx, dvdy = gradUV_stencil_2D(u, v, i, j, dx, dy)
                grad_vel = np.array([[dudx, dudy], [dvdx, dvdy]])
                S = 0.5 * (grad_vel + grad_vel.T)
                ile[i, j] = eigvalsh_max_2D(S)

    return ile


@njit(parallel=True)
def S_eig_2D_data(u, v, dx, dy, mask=None):
    """
    Compute eigenvalues and eigenvectors of Eulerian rate of strain tensor in 2D from u,v which
    describe the x,y velocity.

    Parameters
    ----------
    u : np.ndarray, shape = (nx, ny)
        array containing velocity values in x-direction.
    v : np.ndarray, shape = (nx, ny)
        array containing velocity values in y-direction.
    dx : float
        grid spacing in x-direction.
    dy : float
        grid spacing in y-direction.
    mask : None or np.ndarray, shape = (nx, ny), optional
        for masked data, pass in a boolean mask corresponding to nan values
        (True indicates a nan value). To avoid erroneous computations at mask
        boundaries, mask passed in should be dilated using the
        binary_mask_dilation function from the utils module. The default is None.


    Returns
    -------
    eigvals : np.ndarray, shape = (nx, ny, 2)
        array containg eigenvalues of S.
    eigvecs : np.ndarray, shape = (nx, ny, 2, 2)
        array containing eigenvectors of S.

    """

    nx, ny = u.shape
    eigvals = np.zeros((nx, ny, 2), float64)
    eigvecs = np.zeros((nx, ny, 2, 2), float64)
    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            if mask is None or not mask[i, j]:
                dudx, dudy, dvdx, dvdy = gradUV_stencil_2D(u, v, i, j, dx, dy)
                grad_vel = np.array([[dudx, dudy], [dvdx, dvdy]])
                S = 0.5 * (grad_vel + grad_vel.T)
                evals_tmp, evecs_tmp = np.linalg.eigh(S)
                eigvals[i, j, :] = evals_tmp
                eigvecs[i, j, :, :] = evecs_tmp

    return eigvals, eigvecs


def ivd_grid_2D(vort, vort_avg):
    """
    Compute IVD at an instant from vort and vort_avg.

    Parameters
    ----------
    vort : np.ndarray, shape = (nx, ny)
        array containg vorticity values.
    vort_avg : float
        value of average spatial vorticity at specific time.

    Returns
    -------
    ivd : np.ndarray, shape = (nx, ny)
        array containing values of ivd.

    """

    ivd = np.abs(vort - vort_avg)
    return ivd

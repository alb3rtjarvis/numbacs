from math import sin, cos, pi, cosh, tanh, sqrt, floor, atan2, asin
import numpy as np
from numba import njit, cfunc, float64, guvectorize
from numbalsoda import lsoda_sig
from interpolation.splines import UCGrid, prefilter, eval_spline, eval_linear
from interpolation.splines import extrap_options as xto


def get_interp_arrays_2D(tvals, xvals, yvals, U, V):
    """
    Compute coefficient arrays for cubic spline of velocity field defined by U,V over values
    tvals,xvals,yvals and return the grid tuple and coefficient arrays which can be used
    by 'eval_spline' function of the interpolation package.

    Parameters
    ----------
    tvals : np.ndarray, shape = (nt,)
        times over which the ode is defined, must be ascending.
    xvals : np.ndarray, shape = (nx,)
        x values over which the ode is defined, must be ascending.
    yvals : np.ndarray, shape = (ny,)
        y values over which the ode is defined, must be ascending.
    U : np.ndarray, shape = (nt,nx,ny)
        x-compnent of the velocity.
    V : np.ndarray, shape = (nt,nx,ny)
        y-compnent of the velocity.

    Returns
    -------
    grid_vel : tuple
        grid endpoints and number of points in x and y directions
    C_eval_u : np.ndarray, shape = (nt+2,nx+2,ny+2)
        array containing coefficients for u cubic spline.
    C_eval_v : np.ndarray, shape = (nt+2,nx+2,ny+2)
        array containing coefficients for v cubic spline.

    """

    nt, nx, ny = U.shape
    grid_vel = UCGrid(
        (tvals[0], tvals[-1], nt), (xvals[0], xvals[-1], nx), (yvals[0], yvals[-1], ny)
    )
    C_eval_u = prefilter(grid_vel, U, out=None, k=3)
    C_eval_v = prefilter(grid_vel, V, out=None, k=3)

    return grid_vel, C_eval_u, C_eval_v


def get_interp_arrays_2D_steady(xvals, yvals, U, V):
    """
    Compute coefficient arrays for cubic spline of velocity field defined by U,V over values
    xvals,yvals and return the grid tuple and coefficient arrays which can be used
    by 'eval_spline' function of the interpolation package.

    Parameters
    ----------
    xvals : np.ndarray, shape = (nx,)
        x values over which the ode is defined, must be ascending.
    yvals : np.ndarray, shape = (ny,)
        y values over which the ode is defined, must be ascending.
    U : np.ndarray, shape = (nt,nx,ny)
        x-compnent of the velocity.
    V : np.ndarray, shape = (nt,nx,ny)
        y-compnent of the velocity.

    Returns
    -------
    grid_vel : tuple
        grid endpoints and number of points in x and y directions
    C_eval_u : np.ndarray, shape = (nt+2,nx+2,ny+2)
        array containing coefficients for u cubic spline.
    C_eval_v : np.ndarray, shape = (nt+2,nx+2,ny+2)
        array containing coefficients for v cubic spline.

    """

    nx, ny = U.shape
    grid_vel = UCGrid((xvals[0], xvals[-1], nx), (yvals[0], yvals[-1], ny))
    C_eval_u = prefilter(grid_vel, U, out=None, k=3)
    C_eval_v = prefilter(grid_vel, V, out=None, k=3)

    return grid_vel, C_eval_u, C_eval_v


def get_interp_arrays_scalar(tvals, xvals, yvals, f):
    """
    Compute coefficient arrays for cubic spline of scalar field f defined over values
    tvals,xvals,yvals and return the grid tuple and coefficient array which can be used
    by 'eval_spline' function of the interpolation package.

    Parameters
    ----------
    tvals : np.ndarray, shape = (nt,)
        times over which the f is defined, must be ascending.
    xvals : np.ndarray, shape = (nx,)
        x values over which f ode is defined, must be ascending.
    yvals : np.ndarray, shape = (ny,)
        y values over which the f is defined, must be ascending.
    f : np.ndarray, shape = (nt,nx,ny)
        scalar value to be interpolated.

    Returns
    -------
    grid_f : tuple
        grid endpoints and number of points in t, x and y directions
    C_eval_f : np.ndarray, shape = (nt+2,nx+2,ny+2)
        array containing coefficients for f cubic spline.

    """

    nt, nx, ny = f.shape
    if tvals[1] < tvals[0]:
        f = np.flip(f, axis=0)
        tvals = tvals[::-1]
    grid_f = UCGrid((tvals[0], tvals[-1], nt), (xvals[0], xvals[-1], nx), (yvals[0], yvals[-1], ny))
    C_eval_f = prefilter(grid_f, f, out=None, k=3)

    return grid_f, C_eval_f


def get_flow_2D(grid_vel, C_eval_u, C_eval_v, spherical=0, extrap_mode="constant", r=6371.0):
    """
    Create a C callback for the ode defined by the vector field (U,V) defined over
    a spatial grid given by (xvals,yvals) over times defined by tvals. Cubic interpolant
    is used.

    Parameters
    ----------
    grid_vel : tuple
        grid endpoints and number of points in x and y directions
    C_eval_u : np.ndarray, shape = (nt+2,nx+2,ny+2)
        array containing coefficients for u cubic spline.
    C_eval_v : np.ndarray, shape = (nt+2,nx+2,ny+2)
        array containing coefficients for v cubic spline.
    spherical : int, optional
        int used to determine if flow is defined in spherical coordinate system;
        0 if not spherical,
        1 if spherical and lon = [-180,180),
        2 if spherical and lon = [0,360),
        units expected are degrees and both lon and lat must be ascending,
        lat is expected to = [-90,90]. The default is 0.
    extrap_mode : str, optional
        type of extrapolation mode used for interpolant. The default is 'constant'.
    r : float, optional
        radius used for spherical conversion, used if spherical > 0. The default is 6371.

    Returns
    -------
    funcptr : int
        address to C callback.

    """

    if spherical == 1:

        @cfunc(lsoda_sig)
        def flow_rhs(t, y, dy, p):
            """
            p[0] = int_direction.
            """
            tt = p[0] * t
            xx = ((y[0] - 180) % 360) - 180
            yy = y[1]
            point = np.array([tt, xx, yy])
            dy[0] = (
                (
                    p[0]
                    * eval_spline(
                        grid_vel,
                        C_eval_u,
                        point,
                        out=None,
                        k=3,
                        diff="None",
                        extrap_mode=extrap_mode,
                    )
                )
                * 180
                / (pi * r * cos(yy * pi / 180))
            )
            dy[1] = (
                (
                    p[0]
                    * eval_spline(
                        grid_vel,
                        C_eval_v,
                        point,
                        out=None,
                        k=3,
                        diff="None",
                        extrap_mode=extrap_mode,
                    )
                )
                * 180
                / (pi * r)
            )
    elif spherical == 2:

        @cfunc(lsoda_sig)
        def flow_rhs(t, y, dy, p):
            """
            p[0] = int_direction.
            """
            tt = p[0] * t
            xx = y[0] % 360
            yy = y[1]
            point = np.array([tt, xx, yy])
            dy[0] = (
                (
                    p[0]
                    * eval_spline(
                        grid_vel,
                        C_eval_u,
                        point,
                        out=None,
                        k=3,
                        diff="None",
                        extrap_mode=extrap_mode,
                    )
                )
                * 180
                / (pi * r * cos(yy * pi / 180))
            )
            dy[1] = (
                (
                    p[0]
                    * eval_spline(
                        grid_vel,
                        C_eval_v,
                        point,
                        out=None,
                        k=3,
                        diff="None",
                        extrap_mode=extrap_mode,
                    )
                )
                * 180
                / (pi * r)
            )
    else:

        @cfunc(lsoda_sig)
        def flow_rhs(t, y, dy, p):
            """
            p[0] = int_direction.
            """
            tt = p[0] * t
            point = np.array([tt, y[0], y[1]])
            dy[0] = p[0] * eval_spline(
                grid_vel, C_eval_u, point, out=None, k=3, diff="None", extrap_mode=extrap_mode
            )
            dy[1] = p[0] * eval_spline(
                grid_vel, C_eval_v, point, out=None, k=3, diff="None", extrap_mode=extrap_mode
            )

    funcptr = flow_rhs.address

    return funcptr


def get_callable_2D(
    grid_vel, C_eval_u, C_eval_v, spherical=0, extrap_mode="constant", r=6371.0, return_type="array"
):
    """
    Create a jit-callable spline for the ode defined by the vector field (U,V) defined over
    a spatial grid given by (xvals,yvals) over times defined by tvals.

    Parameters
    ----------
    grid_vel : tuple
        grid endpoints and number of points in t, x, and y directions
    C_eval_u : np.ndarray, shape = (nt+2,nx+2,ny+2)
        array containing coefficients for u cubic spline.
    C_eval_v : np.ndarray, shape = (nt+2,nx+2,ny+2)
        array containing coefficients for v cubic spline.
    spherical : int, optional
        int used to determine if flow is defined in spherical coordinate system;
        0 if not spherical,
        1 if spherical,
        units expected are degrees and both lon and lat must be ascending,
        lat is expected to = [-90,90]. The default is 0.
    extrap_mode : str, optional
        type of extrapolation mode used for interpolant. The default is 'constant'.
    r : float, optional
        radius used for spherical conversion, used if spherical = 1. The default is 6371.

    Returns
    -------
    vel_spline : jit-callable
        jit-callable function for vector field.

    """

    if spherical == 1:
        if return_type == "array":

            @njit
            def vel_spline(point):
                ui = (
                    eval_spline(
                        grid_vel,
                        C_eval_u,
                        point,
                        out=None,
                        k=3,
                        diff="None",
                        extrap_mode=extrap_mode,
                    )
                    * 180
                    / (pi * r * cos(point[2] * pi / 180))
                )
                vi = (
                    eval_spline(
                        grid_vel,
                        C_eval_v,
                        point,
                        out=None,
                        k=3,
                        diff="None",
                        extrap_mode=extrap_mode,
                    )
                    * 180
                    / (pi * r)
                )
                return np.array([ui, vi], float64)

        elif return_type == "tuple":

            @njit
            def vel_spline(point):
                ui = (
                    eval_spline(
                        grid_vel,
                        C_eval_u,
                        point,
                        out=None,
                        k=3,
                        diff="None",
                        extrap_mode=extrap_mode,
                    )
                    * 180
                    / (pi * r * cos(point[2] * pi / 180))
                )
                vi = (
                    eval_spline(
                        grid_vel,
                        C_eval_v,
                        point,
                        out=None,
                        k=3,
                        diff="None",
                        extrap_mode=extrap_mode,
                    )
                    * 180
                    / (pi * r)
                )
                return ui, vi

    else:
        if return_type == "array":

            @njit
            def vel_spline(point):
                ui = eval_spline(
                    grid_vel, C_eval_u, point, out=None, k=3, diff="None", extrap_mode=extrap_mode
                )
                vi = eval_spline(
                    grid_vel, C_eval_v, point, out=None, k=3, diff="None", extrap_mode=extrap_mode
                )
                return np.array([ui, vi], float64)

        elif return_type == "tuple":

            @njit
            def vel_spline(point):
                ui = eval_spline(
                    grid_vel, C_eval_u, point, out=None, k=3, diff="None", extrap_mode=extrap_mode
                )
                vi = eval_spline(
                    grid_vel, C_eval_v, point, out=None, k=3, diff="None", extrap_mode=extrap_mode
                )
                return ui, vi

    return vel_spline


def get_callable_scalar(grid_f, C_eval_f, extrap_mode="constant"):
    """
    Create jit-callable spline for scalar field f defined over grid_f.

    Parameters
    ----------
    grid_f : tuple
        grid endpoints and number of points in t, x, and y directions.
    C_eval_f : np.ndarray, shape = (nt+2,nx+2,ny+2)
        array containing coefficients for f cubic spline.
    extrap_mode : str, optional
        type of extrapolation mode used for interpolant. The default is 'constant'.

    Returns
    -------
    f_spline : jit-callable
        jit callable function for spline of f.

    """

    @njit
    def f_spline(point):
        fi = eval_spline(
            grid_f, C_eval_f, point, out=None, k=3, diff="None", extrap_mode=extrap_mode
        )

        return fi

    return f_spline


def get_flow_linear_2D(grid_vel, U, V, spherical=0, extrap_mode="constant", r=6371.0):
    """
    Create a C callback for the ode defined by the vector field (U,V) defined over
    a grid given by (xvals,yvals) over times defined by tvals. Linear interpolant
    is used. It is currently recommended to use the spline version as, in our tests,
    the spline is both more accurate and faster when used in ode solver.

    Parameters
    ----------
    grid_vel : tuple
        grid endpoints and number of points in x and y directions
    U : np.ndarray, shape = (nt,nx,ny)
        x-compnent of the velocity.
    V : np.ndarray, shape = (nt,nx,ny)
        y-compnent of the velocity.
    spherical : int, optional
        int used to determine if flow is defined in spherical coordinate system;
        0 if not spherical,
        1 if spherical and lon = [-180,180),
        2 if spherical and lon = [0,360),
        units expected are degrees and both lon and lat must be ascending,
        lat is expected to = [-90,90]. The default is 0.
    extrap_mode : str, optional
        type of extrapolation mode used for interpolant. The default is 'constant'.
    r : float, optional
        radius used for spherical conversion, used if spherical > 0. The default is 6371.

    Returns
    -------
    funcptr : int
        address to C callback.

    """
    if extrap_mode == "constant":
        extrap_mode = xto.CONSTANT
    elif extrap_mode == "linear":
        extrap_mode = xto.LINEAR
    elif extrap_mode == "nearest":
        extrap_mode = xto.NEAREST

    if spherical == 1:

        @cfunc(lsoda_sig)
        def flow_rhs(t, y, dy, p):
            """
            p[0] = int_direction.
            """
            tt = p[0] * t
            xx = ((y[0] - 180) % 360) - 180
            yy = y[1]
            point = np.array([tt, xx, yy])
            dy[0] = (
                (p[0] * eval_linear(grid_vel, U, point, extrap_mode))
                * 180
                / (pi * r * cos(yy * pi / 180))
            )
            dy[1] = (p[0] * eval_linear(grid_vel, V, point, extrap_mode)) * 180 / (pi * r)
    elif spherical == 2:

        @cfunc(lsoda_sig)
        def flow_rhs(t, y, dy, p):
            """
            p[0] = int_direction.
            """
            tt = p[0] * t
            xx = y[0] % 360
            yy = y[1]
            point = np.array([tt, xx, yy])
            dy[0] = (
                (p[0] * eval_linear(grid_vel, U, point, extrap_mode))
                * 180
                / (pi * r * cos(yy * pi / 180))
            )
            dy[1] = (p[0] * eval_linear(grid_vel, V, point, extrap_mode)) * 180 / (pi * r)
    else:

        @cfunc(lsoda_sig)
        def flow_rhs(t, y, dy, p):
            """
            p[0] = int_direction.
            """
            tt = p[0] * t
            point = np.array([tt, y[0], y[1]])
            dy[0] = p[0] * eval_linear(grid_vel, U, point, extrap_mode)
            dy[1] = p[0] * eval_linear(grid_vel, V, point, extrap_mode)

    funcptr = flow_rhs.address

    return funcptr


def get_callable_linear_2D(
    grid_vel, U, V, spherical=0, extrap_mode="constant", r=6371.0, return_type="array"
):
    """
    Create a jit-callable for the ode defined by the vector field (U,V) defined over
    a spatial grid given by (xvals,yvals) over times defined by tvals. Linear interpolant
    is used.

    Parameters
    ----------
    grid_vel : tuple
        grid endpoints and number of points in x and y directions
    U : np.ndarray, shape = (nt,nx,ny)
        x-compnent of the velocity.
    V : np.ndarray, shape = (nt,nx,ny)
        y-compnent of the velocity.
    spherical : int, optional
        int used to determine if flow is defined in spherical coordinate system;
        0 if not spherical,
        1 if spherical
        units expected are degrees and both lon and lat must be ascending,
        lat is expected to = [-90,90]. The default is 0.
    return_spline : boolean, optional
        flag to determine if spline is returned. The default is False.
    extrap_mode : str, optional
        type of extrapolation mode used for interpolant. The default is 'constant'.
    r : float, optional
        radius used for spherical conversion, used if spherical > 0. The default is 6371.
    return_type : np.ndarray or tuple, optional
        determines the return type of the callable generated here. If return_type == 'array',
        only a single point can be passed to vel_func
        (needed for funcs from numbacs.diagnostics), if return_type == 'tuple', many points
        (or a single point) can be passed to vel_func at once for quicker computation.

    Returns
    -------
    vel_spline : jit-callable
        jit-callable function for vector field.

    """
    if extrap_mode == "constant":
        extrap_mode = xto.CONSTANT
    elif extrap_mode == "linear":
        extrap_mode = xto.LINEAR
    elif extrap_mode == "nearest":
        extrap_mode = xto.NEAREST

    if spherical == 1:
        if return_type == "array":

            @njit
            def vel_spline(point):
                ui = (
                    eval_linear(grid_vel, U, point, extrap_mode)
                    * 180
                    / (pi * r * cos(point[2] * pi / 180))
                )
                vi = eval_linear(grid_vel, V, point, extrap_mode) * 180 / (pi * r)
                return np.array([ui, vi], float64)

        elif return_type == "tuple":

            @njit
            def vel_spline(point):
                ui = (
                    eval_linear(grid_vel, U, point, extrap_mode)
                    * 180
                    / (pi * r * cos(point[2] * pi / 180))
                )
                vi = eval_linear(grid_vel, V, point, extrap_mode) * 180 / (pi * r)
                return ui, vi

    else:
        if return_type == "array":

            @njit
            def vel_spline(point):
                ui = eval_linear(grid_vel, U, point, extrap_mode)
                vi = eval_linear(grid_vel, V, point, extrap_mode)
                return np.array([ui, vi], float64)

        elif return_type == "tuple":

            @njit
            def vel_spline(point):
                ui = eval_linear(grid_vel, U, point, extrap_mode)
                vi = eval_linear(grid_vel, V, point, extrap_mode)
                return ui, vi

    return vel_spline


def get_callable_scalar_linear(grid_f, f, extrap_mode="constant"):
    """
    Create jit-callable linear interpolant for scalar field f defined over grid_f.
    Linear interpolant is used.

    Parameters
    ----------
    grid_f : tuple
        grid endpoints and number of points in t, x, and y directions.
    f : np.ndarray, shape = (nt,nx,ny)
        array containing values of f.
    extrap_mode : str, optional
        type of extrapolation mode used for interpolant. The default is 'constant'.

    Returns
    -------
    f_interp : jit-callable
        jit callable function for linear interpolant of f.

    """

    if extrap_mode == "constant":
        extrap_mode = xto.CONSTANT
    elif extrap_mode == "linear":
        extrap_mode = xto.LINEAR
    elif extrap_mode == "nearest":
        extrap_mode = xto.NEAREST

    @njit
    def f_interp(point):
        fi = eval_linear(grid_f, f, point, extrap_mode)

        return fi

    return f_interp


@njit(inline="always")
def _bilinear_interp(f, i0, i1, j0, j1, wx, wy):
    """Bilinear interpolation."""

    # get grid values
    f00 = f[i0, j0]
    f01 = f[i0, j1]
    f10 = f[i1, j0]
    f11 = f[i1, j1]

    # top and bottom interpolation, return vertical interp
    fi_bottom = f00 * (1.0 - wx) + f10 * wx
    fi_top = f01 * (1.0 - wx) + f11 * wx

    return fi_bottom * (1.0 - wy) + fi_top * wy


@njit(inline="always")
def _north_polar_interp(f, i0, i1, j_ring, j_pole, wx, wy_polar):
    """Bilinear interp in north polar region."""
    # get ring and pole values
    f_ring0 = f[i0, j_ring]
    f_ring1 = f[i1, j_ring]
    f_pole = f[0, j_pole]

    # interp along ring, return vertical interp
    f_ring = f_ring0 * (1.0 - wx) + f_ring1 * wx

    return f_ring * (1.0 - wy_polar) + f_pole * wy_polar


@njit(inline="always")
def _south_polar_interp(f, i0, i1, j_ring, j_pole, wx, wy_polar):
    """Bilinear interp in south polar region."""

    # get ring and pole values
    f_ring0 = f[i0, j_ring]
    f_ring1 = f[i1, j_ring]
    f_pole = f[0, j_pole]

    # interp along ring, return vertical interp
    f_ring = f_ring0 * (1.0 - wx) + f_ring1 * wx

    return f_pole * (1.0 - wy_polar) + f_ring * wy_polar


@njit
def _globe_spatial_interp(lon_rad, lat_rad, lonlat_grid, vx, vy, vz):
    """Spatial interp on sphere with both poles included."""
    lon0, dlon, nlon = lonlat_grid[0]
    lat0, dlat, nlat = lonlat_grid[1]

    i_float = ((lon_rad - lon0) / dlon) % nlon
    j_float = (lat_rad - lat0) / dlat

    i0 = floor(i_float)
    i1 = (i0 + 1) % nlon

    wx = i_float - i0

    if j_float >= nlat - 2:
        # set lat indices
        j_ring = nlat - 2
        j_pole = nlat - 1

        # calculate y weight
        wy_polar = j_float - j_ring

        # perform interp for each vel component
        vx_interp = _north_polar_interp(vx, i0, i1, j_ring, j_pole, wx, wy_polar)
        vy_interp = _north_polar_interp(vy, i0, i1, j_ring, j_pole, wx, wy_polar)
        vz_interp = _north_polar_interp(vz, i0, i1, j_ring, j_pole, wx, wy_polar)

    elif j_float <= 1.0:
        # set lat indices
        j_pole = 0
        j_ring = 1

        # calculate y weight
        wy_polar = j_float - j_pole

        # perform interp for each vel component
        vx_interp = _south_polar_interp(vx, i0, i1, j_ring, j_pole, wx, wy_polar)
        vy_interp = _south_polar_interp(vy, i0, i1, j_ring, j_pole, wx, wy_polar)
        vz_interp = _south_polar_interp(vz, i0, i1, j_ring, j_pole, wx, wy_polar)

    else:
        # calculate y indices
        j0 = floor(j_float)
        j1 = int(j0 + 1)

        # calculate y weight
        wy = j_float - j0

        # perform interp for each vel component
        vx_interp = _bilinear_interp(vx, i0, i1, j0, j1, wx, wy)
        vy_interp = _bilinear_interp(vy, i0, i1, j0, j1, wx, wy)
        vz_interp = _bilinear_interp(vz, i0, i1, j0, j1, wx, wy)

    return vx_interp, vy_interp, vz_interp


@njit
def _globe_spatial_interp_northpole(lon_rad, lat_rad, lonlat_grid, vx, vy, vz):
    """Spatial interp on sphere with only north pole included."""
    lon0, dlon, nlon = lonlat_grid[0]
    lat0, dlat, nlat = lonlat_grid[1]

    i_float = ((lon_rad - lon0) / dlon) % nlon
    j_float = (lat_rad - lat0) / dlat

    i0 = floor(i_float)
    i1 = (i0 + 1) % nlon

    wx = i_float - i0

    if j_float >= nlat - 2:
        # set lat indices
        j_ring = nlat - 2
        j_pole = nlat - 1

        # calculate y weight
        wy_polar = j_float - j_ring

        # perform interp for each vel component
        vx_interp = _north_polar_interp(vx, i0, i1, j_ring, j_pole, wx, wy_polar)
        vy_interp = _north_polar_interp(vy, i0, i1, j_ring, j_pole, wx, wy_polar)
        vz_interp = _north_polar_interp(vz, i0, i1, j_ring, j_pole, wx, wy_polar)

    else:
        # calculate y indices
        j0 = floor(j_float)
        j1 = int(j0 + 1)

        # calculate y weight
        wy = j_float - j0

        # perform interp for each vel component
        vx_interp = _bilinear_interp(vx, i0, i1, j0, j1, wx, wy)
        vy_interp = _bilinear_interp(vy, i0, i1, j0, j1, wx, wy)
        vz_interp = _bilinear_interp(vz, i0, i1, j0, j1, wx, wy)

    return vx_interp, vy_interp, vz_interp


@njit
def _globe_spatial_interp_southpole(lon_rad, lat_rad, lonlat_grid, vx, vy, vz):
    """Spatial interp on sphere with only north pole included."""
    lon0, dlon, nlon = lonlat_grid[0]
    lat0, dlat, nlat = lonlat_grid[1]

    i_float = ((lon_rad - lon0) / dlon) % nlon
    j_float = (lat_rad - lat0) / dlat

    i0 = floor(i_float)
    i1 = (i0 + 1) % nlon

    wx = i_float - i0

    if j_float <= 1.0:
        # set lat indices
        j_pole = 0
        j_ring = 1

        # calculate y weight
        wy_polar = j_float - j_pole

        # perform interp for each vel component
        vx_interp = _south_polar_interp(vx, i0, i1, j_ring, j_pole, wx, wy_polar)
        vy_interp = _south_polar_interp(vy, i0, i1, j_ring, j_pole, wx, wy_polar)
        vz_interp = _south_polar_interp(vz, i0, i1, j_ring, j_pole, wx, wy_polar)

    else:
        # calculate y indices
        j0 = floor(j_float)
        j1 = int(j0 + 1)

        # calculate y weight
        wy = j_float - j0

        # perform interp for each vel component
        vx_interp = _bilinear_interp(vx, i0, i1, j0, j1, wx, wy)
        vy_interp = _bilinear_interp(vy, i0, i1, j0, j1, wx, wy)
        vz_interp = _bilinear_interp(vz, i0, i1, j0, j1, wx, wy)

    return vx_interp, vy_interp, vz_interp


@njit
def globe_interp(t, y, r, grid, vx, vy, vz):
    """
    Linear interpolation on the sphere with both poles included.

    Parameters
    ----------
    t : float
        time.
    y : np.ndarray, shape=(3,)
        DESCRIPTION.
    r : float
        raidus of sphere in same units as y.
    grid : tuple
        tuple containing min value, spacing, and num points, for t and
        underlying lon-lat grid.
    vx : np.ndarray, shape=(nt, nx, ny)
        velocity in the x direction.
    vy : np.ndarray, shape=(nt, nx, ny)
        velocity in the y direction.
    vz : np.ndarray, shape=(nt, nx, ny)
        velocity in the z direction.

    Returns
    -------
    np.ndarray, shape=(3,)
        x, y, z velocity at (t, y).

    """

    t0, dt, nt = grid[0]
    lonlat_grid = grid[1:]

    k_float = (t - t0) / dt
    k0 = floor(k_float)

    xk, yk, zk = y[0], y[1], y[2]
    lon_rad = atan2(yk, xk)

    z_ratio = zk / r
    if z_ratio >= 1.0:
        lat_rad = pi / 2
    elif z_ratio <= -1.0:
        lat_rad = -pi / 2
    else:
        lat_rad = asin(z_ratio)

    if k0 >= nt - 1:
        return np.array(
            _globe_spatial_interp(lon_rad, lat_rad, lonlat_grid, vx[-1], vy[-1], vz[-1])
        )

    else:
        k1 = k0 + 1
        wt = k_float - k0
        vk0 = _globe_spatial_interp(lon_rad, lat_rad, lonlat_grid, vx[k0], vy[k0], vz[k0])
        vk1 = _globe_spatial_interp(lon_rad, lat_rad, lonlat_grid, vx[k1], vy[k1], vz[k1])

        vx_interp = (1.0 - wt) * vk0[0] + wt * vk1[0]
        vy_interp = (1.0 - wt) * vk0[1] + wt * vk1[1]
        vz_interp = (1.0 - wt) * vk0[2] + wt * vk1[2]

        return np.array([vx_interp, vy_interp, vz_interp])


@njit
def globe_interp_northpole(t, y, r, grid, vx, vy, vz):
    """
    Linear interpolation on the sphere with only north pole included.

    Parameters
    ----------
    t : float
        time.
    y : np.ndarray, shape=(3,)
        DESCRIPTION.
    r : float
        raidus of sphere in same units as y.
    grid : tuple
        tuple containing min value, spacing, and num points, for t and
        underlying lon-lat grid.
    vx : np.ndarray, shape=(nt, nx, ny)
        velocity in the x direction.
    vy : np.ndarray, shape=(nt, nx, ny)
        velocity in the y direction.
    vz : np.ndarray, shape=(nt, nx, ny)
        velocity in the z direction.

    Returns
    -------
    np.ndarray, shape=(3,)
        x, y, z velocity at (t, y).

    """
    t0, dt, nt = grid[0]
    lonlat_grid = grid[1:]

    k_float = (t - t0) / dt
    k0 = floor(k_float)

    xk, yk, zk = y[0], y[1], y[2]
    lon_rad = atan2(yk, xk)

    z_ratio = zk / r
    if z_ratio >= 1.0:
        lat_rad = pi / 2
    elif z_ratio <= -1.0:
        lat_rad = -pi / 2
    else:
        lat_rad = asin(z_ratio)

    if k0 >= nt - 1:
        return np.array(
            _globe_spatial_interp_northpole(lon_rad, lat_rad, lonlat_grid, vx[-1], vy[-1], vz[-1])
        )

    else:
        k1 = k0 + 1
        wt = k_float - k0
        vk0 = _globe_spatial_interp_northpole(lon_rad, lat_rad, lonlat_grid, vx[k0], vy[k0], vz[k0])
        vk1 = _globe_spatial_interp_northpole(lon_rad, lat_rad, lonlat_grid, vx[k1], vy[k1], vz[k1])

        vx_interp = (1.0 - wt) * vk0[0] + wt * vk1[0]
        vy_interp = (1.0 - wt) * vk0[1] + wt * vk1[1]
        vz_interp = (1.0 - wt) * vk0[2] + wt * vk1[2]

        return np.array([vx_interp, vy_interp, vz_interp])


@njit
def globe_interp_southpole(t, y, r, grid, vx, vy, vz):
    """
    Linear interpolation on the sphere with only south pole included.

    Parameters
    ----------
    t : float
        time.
    y : np.ndarray, shape=(3,)
        DESCRIPTION.
    r : float
        raidus of sphere in same units as y.
    grid : tuple
        tuple containing min value, spacing, and num points, for t and
        underlying lon-lat grid.
    vx : np.ndarray, shape=(nt, nx, ny)
        velocity in the x direction.
    vy : np.ndarray, shape=(nt, nx, ny)
        velocity in the y direction.
    vz : np.ndarray, shape=(nt, nx, ny)
        velocity in the z direction.

    Returns
    -------
    np.ndarray, shape=(3,)
        x, y, z velocity at (t, y).

    """
    t0, dt, nt = grid[0]
    lonlat_grid = grid[1:]

    k_float = (t - t0) / dt
    k0 = floor(k_float)

    xk, yk, zk = y[0], y[1], y[2]
    lon_rad = atan2(yk, xk)

    z_ratio = zk / r
    if z_ratio >= 1.0:
        lat_rad = pi / 2
    elif z_ratio <= -1.0:
        lat_rad = -pi / 2
    else:
        lat_rad = asin(z_ratio)

    if k0 >= nt - 1:
        return np.array(
            _globe_spatial_interp_southpole(lon_rad, lat_rad, lonlat_grid, vx[-1], vy[-1], vz[-1])
        )

    else:
        k1 = k0 + 1
        wt = k_float - k0
        vk0 = _globe_spatial_interp_southpole(lon_rad, lat_rad, lonlat_grid, vx[k0], vy[k0], vz[k0])
        vk1 = _globe_spatial_interp_southpole(lon_rad, lat_rad, lonlat_grid, vx[k1], vy[k1], vz[k1])

        vx_interp = (1.0 - wt) * vk0[0] + wt * vk1[0]
        vy_interp = (1.0 - wt) * vk0[1] + wt * vk1[1]
        vz_interp = (1.0 - wt) * vk0[2] + wt * vk1[2]

        return np.array([vx_interp, vy_interp, vz_interp])


def get_globe_flow(tvals, lonvals, latvals, vx, vy, vz, r=6371.0, pole="both"):
    """
    Retrieve address of cfunc represntation of linear interpolant on the sphere.

    Parameters
    ----------
    tvals : np.ndarray, shape=(nx,)
        time values.
    lonvals : np.ndarray, shape=(nt,)
        longitude values.
    latvals : np.ndarray, shape=(nt,)
        latitude values.
    vx : np.ndarray, shape=(nt, nx, ny)
        velocity in the x direction.
    vy : np.ndarray, shape=(nt, nx, ny)
        velocity in the y direction.
    vz : np.ndarray, shape=(nt, nx, ny)
        velocity in the z direction.
    r : float, optional
        raidus of sphere the flow is defined on, defaults to radius of the earth (in km).
        The default is 6371.0.
    pole : str, optional
        str to determine if any poles are included in the velocity fields. Options are
        "both", "north", or "south". The default is "both".

    Raises
    ------
    ValueError
        if no poles are included, use regular linear interpolant.

    Returns
    -------
    int
        address for C callback.

    """
    t0 = tvals[0]
    dt = tvals[1] - t0
    nt = len(tvals)

    lon0 = lonvals[0]
    dlon = lonvals[1] - lon0
    nlon = len(lonvals)

    lat0 = latvals[0]
    dlat = latvals[1] - lat0
    nlat = len(latvals)

    grid = ((t0, dt, nt), (lon0, dlon, nlon), (lat0, dlat, nlat))

    pole = pole.lower()
    if pole == "both":
        f = globe_interp
    elif pole == "north":
        f = globe_interp_northpole
    elif pole == "south":
        f = globe_interp_southpole
    else:
        raise ValueError("Argument 'pole' should be 'both', 'north', or 'south'.")

    @cfunc(lsoda_sig)
    def flow_rhs(t, y, dy, p):
        int_direction = p[0]
        tt = int_direction * t
        vxi, vyi, vzi = f(tt, y, r, grid, vx, vy, vz)

        # norm_x, norm_y, norm_z = y / r
        inv_r = 1 / r
        norm_x = y[0] * inv_r
        norm_y = y[1] * inv_r
        norm_z = y[2] * inv_r

        radial = vxi * norm_x + vyi * norm_y + vzi * norm_z

        vx_tan = vxi - radial * norm_x
        vy_tan = vyi - radial * norm_y
        vz_tan = vzi - radial * norm_z

        dy[0] = int_direction * vx_tan
        dy[1] = int_direction * vy_tan
        dy[2] = int_direction * vz_tan

    return flow_rhs.address


def get_predefined_flow(
    flow_str,
    int_direction=1.0,
    return_default_params=True,
    return_domain=True,
    parameter_description=False,
):
    """
    Create a C callback for one of the predefined flows.

    Parameters
    ----------
    flow_str : str
        string representing which flow to retrieve. Currently 'double_gyre', 'bickley_jet',
        and 'abc' are supported.
    int_direction : float, optional
        direction of integration (either -1. or 1.). The default is 1.
    return_default_params : boolean, optional
        flag to determine if default parameters will be returned. The default is True.
    return_domain : boolean, optional
        flag to determine if domain will be returned. The default is True.
    parameter_description : boolean, optional
        flag to determine if string containing description of parameters is returned.
        The default if False.

    Returns
    -------
    funcptr : int
        address of C callback.
    default_params : np.ndarray, shape = (nprms,), optional
        default parameters.
    domain : tuple, optional
        array containing endpoints of domain for each dimension.
    p_str : str, optional
        string containing description of parameters in equation.


    """

    match flow_str:
        case "double_gyre":

            @cfunc(lsoda_sig)
            def _double_gyre(t, y, dy, p):
                """
                p[0] = int_direction, p[1] = A, p[2] = eps, p[3] = alpha, p[4] = omega,
                p[5] = psi.
                """
                tt = p[0] * t
                a = p[2] * sin(p[4] * tt + p[5])
                b = 1 - 2 * a
                f = a * y[0] ** 2 + b * y[0]
                df = 2 * a * y[0] + b
                dy[0] = p[0] * (-pi * p[1] * sin(pi * f) * cos(pi * y[1]) - p[3] * y[0])
                dy[1] = p[0] * (pi * p[1] * cos(pi * f) * sin(pi * y[1]) * df - p[3] * y[1])

            funcptr = _double_gyre.address

            if return_default_params:
                A = 0.1
                eps = 0.25
                alpha = 0.0
                omega = 0.2 * pi
                psi = 0.0

                default_params = np.array([int_direction, A, eps, alpha, omega, psi])

            if return_domain:
                domain = ((0.0, 2.0), (0.0, 1.0))

            if parameter_description:
                p_str = (
                    "p[0] = int_direction, p[1] = A, p[2] = eps, p[3] = alpha, "
                    + "p[4] = omega, p[5] = psi"
                )

        case "bickley_jet":

            @cfunc(lsoda_sig)
            def _bickley_jet(t, y, dy, p):
                """
                p[0] = int_direction, p[1] = U0, p[2] = L, p[3] = A1, p[4] = A2,
                p[5] = A3, p[6] = k1, p[7] = k2, p[8] = k3, p[9] = c1, p[10] = c2, p[11] = c3.
                """

                tt = p[0] * t
                Y = y[1] / p[2]
                sech2 = 1 / (cosh(Y) ** 2)
                dy[0] = p[0] * (
                    p[1] * sech2
                    + 2
                    * p[1]
                    * tanh(Y)
                    * sech2
                    * (
                        p[3] * cos(p[6] * (y[0] - p[9] * tt))
                        + p[4] * cos(p[7] * (y[0] - p[10] * tt))
                        + p[5] * cos(p[8] * (y[0] - p[11] * tt))
                    )
                )
                dy[1] = -p[0] * (
                    p[1]
                    * p[2]
                    * sech2
                    * (
                        p[3] * p[6] * sin(p[6] * (y[0] - p[9] * tt))
                        + p[4] * p[7] * sin(p[7] * (y[0] - p[10] * tt))
                        + p[5] * p[8] * sin(p[8] * (y[0] - p[11] * tt))
                    )
                )

            funcptr = _bickley_jet.address

            if return_default_params:
                # units: time - days, length - Mm
                int_direction = 1.0
                r_e = 6371.0e-3
                U0 = 86400 * 62.66e-6
                L = 1770.0e-3
                A1 = 0.0075
                A2 = 0.15
                A3 = 0.3
                k1 = 2.0 / r_e
                k2 = 4.0 / r_e
                k3 = 6.0 / r_e
                c2 = 0.205 * U0
                c3 = 0.461 * U0
                c1 = c3 + (sqrt(5) - 1) * (c2 - c3)

                default_params = np.array(
                    [int_direction, U0, L, A1, A2, A3, k1, k2, k3, c1, c2, c3]
                )

            if return_domain:
                domain = ((0.0, r_e * pi), (-3.0, 3.0))

            if parameter_description:
                p_str = (
                    "p[0] = int_direction, p[1] = U0, p[2] = L, p[3] = A1, p[4] = A2, "
                    + "p[5] = A3, p[6] = k1, p[7] = k2, p[8] = k3, p[9] = c1, p[10] = c2, "
                    + "p[11] = c3, units: time - days, length - Mm"
                )

        case "abc":

            @cfunc(lsoda_sig)
            def _abc(t, y, dy, p):
                """
                p[0] = int_direction, p[1] = A-amplitude, p[2] = B-amplitude, p[3] = C-amplitude
                p[4] = forcing amplitdue.
                """
                tt = p[0] * t
                dy[0] = p[0] * ((p[1] + p[4] * tt * sin(pi * tt)) * sin(y[2]) + p[3] * cos(y[1]))
                dy[1] = p[0] * (p[2] * sin(y[0]) + (p[1] + p[4] * tt * sin(pi * tt)) * cos(y[1]))
                dy[2] = p[0] * (p[3] * sin(y[1]) + p[2] * cos(y[1]))

            funcptr = _abc.address

            if return_default_params:
                A = 3**0.5
                B = 2**0.5
                C = 1.0
                f = 0.5

                default_params = np.array([int_direction, A, B, C, f])

            if return_domain:
                domain = ((0.0, 2 * pi), (0.0, 2 * pi), (0.0, 2 * pi))

            if parameter_description:
                p_str = (
                    "p[0] = int_direction, p[1] = A-amplitude, p[2] = B-amplitude, "
                    + "p[3] = C-amplitude, p[4] = forcing amplitdue"
                )

    match [return_default_params, return_domain, parameter_description]:
        case [False, False, False]:
            return funcptr
        case [True, False, False]:
            return funcptr, default_params
        case [False, True, False]:
            return funcptr, domain
        case [True, True, False]:
            return funcptr, default_params, domain
        case [False, False, True]:
            return funcptr, p_str
        case [True, False, True]:
            return funcptr, default_params, p_str
        case [False, True, True]:
            return funcptr, domain, p_str
        case [True, True, True]:
            return funcptr, default_params, domain, p_str


def get_predefined_callable(
    flow_str, params=None, return_domain=True, parameter_description=False, return_type="array"
):
    """
    Create a jit-callable for one of the predefined flows.

    Parameters
    ----------
    flow_str : str
        string representing which flow to retrieve. Currently 'double_gyre', 'bickley_jet',
        and 'abc' are supported.
    params : np.ndarray, shape = (nprms,), optional
        parameters to be used to define the flow. The default is None, i.e. default params.
    return_domain : boolean, optional
        flag to determine if domain will be returned. The default is True.
    parameter_description : boolean, optional
        flag to determine if string containing description of parameters is returned.
        The default if False.

    Returns
    -------
    func : jit-callable
        jit callable function for vector field.
    domain : tuple, optional
        array containing endpoints of domain for each dimension.
    p_str : str, optional
        string containing description of parameters in equation.


    """

    match flow_str:
        case "double_gyre":
            if params is None:
                A = 0.1
                eps = 0.25
                alpha = 0.0
                omega = 0.2 * pi
                psi = 0.0
                default_params = np.array([A, eps, alpha, omega, psi])
                p = default_params
            else:
                p = params

            if return_type == "array":

                @njit
                def _double_gyre(y):
                    """
                    p[0] = A, p[1] = eps, p[2] = alpha, p[3] = omega, p[4] = psi.
                    """

                    a = p[1] * sin(p[3] * y[0] + p[4])
                    b = 1 - 2 * a
                    f = a * y[1] ** 2 + b * y[1]
                    df = 2 * a * y[1] + b
                    dx = -pi * p[0] * sin(pi * f) * cos(pi * y[2]) - p[2] * y[1]
                    dy = pi * p[0] * cos(pi * f) * sin(pi * y[2]) * df - p[2] * y[2]

                    return np.array([dx, dy], float64)

            elif return_type == "tuple":

                @guvectorize(
                    ["void(float64[::1],float64[::1],float64[::1])"],
                    "(n)->(),()",
                    nopython=True,
                    target="cpu",
                )
                def _double_gyre(y, dx, dy):
                    """
                    p[0] = A, p[1] = eps, p[2] = alpha, p[3] = omega, p[4] = psi.
                    """

                    a = p[1] * np.sin(p[3] * y[0] + p[4])
                    b = 1 - 2 * a
                    f = a * y[1] ** 2 + b * y[1]
                    df = 2 * a * y[1] + b
                    dx[:] = -pi * p[0] * np.sin(pi * f) * np.cos(pi * y[2]) - p[2] * y[1]
                    dy[:] = pi * p[0] * np.cos(pi * f) * np.sin(pi * y[2]) * df - p[2] * y[2]

            func = _double_gyre

            if return_domain:
                domain = ((0.0, 2.0), (0.0, 1.0))

            if parameter_description:
                p_str = "p[0] = A, p[1] = eps, p[2] = alpha, p[3] = omega, p[4] = psi"

        case "bickley_jet":
            if params is None:
                # units: time - days, length - Mm
                r_e = 6371.0e-3
                U0 = 86400 * 62.66e-6
                L = 1770.0e-3
                A1 = 0.0075
                A2 = 0.15
                A3 = 0.3
                k1 = 2.0 / r_e
                k2 = 4.0 / r_e
                k3 = 6.0 / r_e
                c2 = 0.205 * U0
                c3 = 0.461 * U0
                c1 = c3 + (sqrt(5) - 1) * (c2 - c3)
                default_params = np.array([U0, L, A1, A2, A3, k1, k2, k3, c1, c2, c3])
                p = default_params
            else:
                p = params

            if return_type == "array":

                @njit
                def _bickley_jet(y):
                    """
                    p[0] = U0, p[1] = L, p[2] = A1, p[3] = A2,
                    p[4] = A3, p[5] = k1, p[6] = k2, p[7] = k3, p[8] = c1, p[9] = c2, p[10] = c3.
                    """

                    Y = y[2] / p[1]
                    sech2 = 1 / (cosh(Y) ** 2)
                    dx = p[0] * sech2 + 2 * p[0] * tanh(Y) * sech2 * (
                        p[2] * cos(p[5] * (y[1] - p[8] * y[0]))
                        + p[3] * cos(p[6] * (y[1] - p[9] * y[0]))
                        + p[4] * cos(p[7] * (y[1] - p[10] * y[0]))
                    )
                    dy = (
                        -p[0]
                        * p[1]
                        * sech2
                        * (
                            p[2] * p[5] * sin(p[5] * (y[1] - p[8] * y[0]))
                            + p[3] * p[6] * sin(p[6] * (y[1] - p[9] * y[0]))
                            + p[4] * p[7] * sin(p[7] * (y[1] - p[10] * y[0]))
                        )
                    )

                    return np.array([dx, dy], float64)

            elif return_type == "tuple":

                @guvectorize(
                    ["void(float64[::1],float64[::1],float64[::1])"],
                    "(n)->(),()",
                    nopython=True,
                    target="cpu",
                )
                def _bickley_jet(y, dx, dy):
                    """
                    p[0] = U0, p[1] = L, p[2] = A1, p[3] = A2,
                    p[4] = A3, p[5] = k1, p[6] = k2, p[7] = k3, p[8] = c1, p[9] = c2, p[10] = c3.
                    """

                    Y = y[2] / p[1]
                    sech2 = 1 / (cosh(Y) ** 2)
                    dx[:] = p[0] * sech2 + 2 * p[0] * tanh(Y) * sech2 * (
                        p[2] * cos(p[5] * (y[1] - p[8] * y[0]))
                        + p[3] * cos(p[6] * (y[1] - p[9] * y[0]))
                        + p[4] * cos(p[7] * (y[1] - p[10] * y[0]))
                    )
                    dy[:] = (
                        -p[0]
                        * p[1]
                        * sech2
                        * (
                            p[2] * p[5] * sin(p[5] * (y[1] - p[8] * y[0]))
                            + p[3] * p[6] * sin(p[6] * (y[1] - p[9] * y[0]))
                            + p[4] * p[7] * sin(p[7] * (y[1] - p[10] * y[0]))
                        )
                    )

            func = _bickley_jet

            if return_domain:
                domain = ((0.0, r_e * pi), (-3.0, 3.0))

            if parameter_description:
                p_str = (
                    "p[0] = U0, p[1] = L, p[2] = A1, p[3] = A2, p[4] = A3, p[5] = k1, "
                    + "p[6] = k2, p[7] = k3, p[8] = c1, p[9] = c2, p[10] = c3, "
                    + "units: time - days, length - Mm"
                )

        case "abc":
            if params is None:
                A = 3**0.5
                B = 2**0.5
                C = 1.0
                f = 0.5
                default_params = np.array([A, B, C, f])
                p = default_params
            else:
                p = params

            if return_type == "array":

                @njit
                def _abc(y):
                    """
                    p[0] = A-amplitude, p[1] = B-amplitude, p[2] = C-amplitude,
                    p[3] = forcing amplitude.
                    """
                    dx = (p[0] + p[3] * y[0] * sin(pi * y[0])) * sin(y[3]) + p[2] * cos(y[2])
                    dy = p[1] * sin(y[1]) + (p[0] + p[3] * y[0] * sin(pi * y[0])) * cos(y[2])
                    dz = p[2] * sin(y[2]) + p[1] * cos(y[2])

                    return np.array([dx, dy, dz], float64)

            elif return_type == "tuple":

                @guvectorize(
                    ["void(float64[::1],float64[::1],float64[::1],float64[::1])"],
                    "(n)->(),(),()",
                    nopython=True,
                    target="cpu",
                )
                def _abc(y, dx, dy, dz):
                    """
                    p[0] = A-amplitude, p[1] = B-amplitude, p[2] = C-amplitude,
                    p[3] = forcing amplitude.
                    """
                    dx[:] = (p[0] + p[3] * y[0] * sin(pi * y[0])) * sin(y[3]) + p[2] * cos(y[2])
                    dy[:] = p[1] * sin(y[1]) + (p[0] + p[3] * y[0] * sin(pi * y[0])) * cos(y[2])
                    dz[:] = p[2] * sin(y[2]) + p[1] * cos(y[2])

            func = _abc

            if return_domain:
                domain = ((0.0, 2 * pi), (0.0, 2 * pi), (0.0, 2 * pi))

            if parameter_description:
                p_str = (
                    "p[0] = A-amplitude, p[1] = B-amplitude, p[2] = C-amplitude, "
                    + "p[3] = forcing amplitude"
                )

    match [return_domain, parameter_description]:
        case [False, False]:
            return func
        case [True, False]:
            return func, domain
        case [False, True]:
            return func, p_str
        case [True, True]:
            return func, domain, p_str

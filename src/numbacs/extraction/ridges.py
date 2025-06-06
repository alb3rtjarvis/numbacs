import numpy as np
from numba import njit, prange
import numba
from ..utils import ravel_index
from math import copysign, floor, acos, atan2, pi
from scipy.ndimage import label, generate_binary_structure


@njit(parallel=True)
def ftle_ridge_pts(f, eigvec_max, x, y, sdd_thresh=0.0, percentile=0):
    """
    Compute FTLE ridge points by finding points (with subpixel accuracy) at which:
    ftle > 0 (or percentile of ftle),
    directional derivaitve of ftle (in eigvec_max direction) is 0,
    second directional derivative of ftle (in eigvec_max direction) is greater than
    sdd_thresh (in magnitude).

    Parameters
    ----------
    f : np.ndarray, shape = (nx,ny)
        ftle array.
    eigvec_max : np.ndarray, shape = (nx,ny,2)
        maximum eigenvector of Cauchy Green tensor.
    x : np.ndarray, shape = (nx,)
        array containing x-values.
    y : np.ndarray, shape = (ny,)
        array containing y-values.
    sdd_thresh : float, optional
        threshold for second directional derivative, should be at least 0.
        The default is 0.
    percentile : int, optional
        percentile of ftle used for min allowed value. The default is 0.

    Returns
    -------
    r_pts : np.ndarray, shape = (ridge_bool.sum(),2)
        ridge points.

    """

    nx, ny = eigvec_max.shape[:-1]
    ridge_bool = np.zeros(nx * ny, numba.bool_)
    r_pts = np.zeros((nx * ny, 2), numba.float64)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    shape_arr = np.array([nx, ny], numba.int32)
    if percentile == 0:
        f_min = 0.0
    else:
        f_min = np.percentile(f, percentile)
    for i in prange(2, nx - 2):
        for j in range(2, ny - 2):
            pt = np.array([x[i], y[j]])
            f0 = f[i, j]
            if f0 > f_min:
                fx = (f[i + 1, j] - f[i - 1, j]) / (2 * dx)
                fy = (f[i, j + 1] - f[i, j - 1]) / (2 * dy)
                fxx = (f[i + 1, j] - 2 * f[i, j] + f[i - 1, j]) / (dx**2)
                fyy = (f[i, j + 1] - 2 * f[i, j] + f[i, j - 1]) / (dy**2)
                fxy = (f[i + 1, j + 1] - f[i + 1, j - 1] - f[i - 1, j + 1] + f[i - 1, j - 1]) / (
                    4 * dx * dy
                )

                eigvec_max0 = eigvec_max[i, j, :]
                ex, ey = eigvec_max0

                c2 = ex * (fxx * ex + fxy * ey) + ey * (fxy * ex + fyy * ey)
                if c2 < -sdd_thresh:
                    t = -(fx * ex + fy * ey) / c2
                    if abs(t * ex) <= dx / 2 and abs(t * ey) <= dy / 2:
                        k = ravel_index(np.array([i, j], numba.int32), shape_arr)
                        r_pts[k, :] = pt + np.array([t * ex, t * ey])
                        ridge_bool[k] = True

    return r_pts[ridge_bool, :]


@njit(parallel=True)
def _ftle_ridges(f, eigvec_max, x, y, sdd_thresh=0.0, percentile=0):
    """
    Compute FTLE ridge points by finding points (with subpixel accuracy) at which:
    ftle > 0 (or percentile of ftle),
    directional derivaitve of ftle (in eigvec_max direction) is 0,
    second directional derivative of ftle (in eigvec_max direction) is greater than
    sdd_thresh (in magnitude).

    Additional returns for use in 'ftle_ridges' function.

    Parameters
    ----------
    f : np.ndarray, shape = (nx,ny)
        ftle array.
    eigvec_max : np.ndarray, shape = (nx,ny,2)
        maximum eigenvector of Cauchy Green tensor.
    x : np.ndarray, shape = (nx,)
        array containing x-values.
    y : np.ndarray, shape = (ny,)
        array containing y-values.
    sdd_thresh : float, optional
        threshold for second directional derivative, should be at least 0.
        The default is 0.
    percentile : int, optional
        percentile of ftle used for min allowed value. The default is 0.

    Returns
    -------
    r_pts : np.ndarray, shape = (ridge_bool.sum(),2)
        ridge points.
    ridge_bool : np.ndarray, shape = (nx,ny)
        truth values determining if ridge point is near a grid point.
    nx : int
        number of grid points in x-direction.
    ny : int
        number of grid points in y-direction.

    """

    nx, ny = eigvec_max.shape[:-1]
    ridge_bool = np.zeros((nx, ny), numba.bool_)
    r_pts = np.zeros((nx * ny, 2), numba.float64)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    shape_arr = np.array([nx, ny], numba.int32)
    if percentile == 0:
        f_min = 0.0
    else:
        f_min = np.percentile(f, percentile)
    for i in prange(2, nx - 2):
        for j in range(2, ny - 2):
            pt = np.array([x[i], y[j]])
            f0 = f[i, j]
            if f0 > f_min:
                fx = (f[i + 1, j] - f[i - 1, j]) / (2 * dx)
                fy = (f[i, j + 1] - f[i, j - 1]) / (2 * dy)
                fxx = (f[i + 1, j] - 2 * f[i, j] + f[i - 1, j]) / (dx**2)
                fyy = (f[i, j + 1] - 2 * f[i, j] + f[i, j - 1]) / (dy**2)
                fxy = (f[i + 1, j + 1] - f[i + 1, j - 1] - f[i - 1, j + 1] + f[i - 1, j - 1]) / (
                    4 * dx * dy
                )

                eigvec_max0 = eigvec_max[i, j, :]
                ex, ey = eigvec_max0

                c2 = ex * (fxx * ex + fxy * ey) + ey * (fxy * ex + fyy * ey)
                if c2 < -sdd_thresh:
                    t = -(fx * ex + fy * ey) / c2
                    if abs(t * ex) <= dx / 2 and abs(t * ey) <= dy / 2:
                        k = ravel_index(np.array([i, j], numba.int32), shape_arr)
                        r_pts[k, :] = pt + np.array([t * ex, t * ey])
                        ridge_bool[i, j] = True

    return r_pts, ridge_bool, nx, ny


@njit
def _get_ridges(ridge_pts, inds, nx, ny, min_ridge_pts):
    """
    JIT-function to speed up finding connected ridges.

    Parameters
    ----------
    ridge_pts : np.ndarray, shape = (nrpts,2)
        array containg position of ridge points.
    inds : np.ndarray, shape = (len(inds),)
        array containing indices corresponding to current ridge points.
    nx : int
        number of grid points in x-direction.
    ny : int
        number of grid points in y-direction.
    min_ridge_pts : int
        minimum points allowed in ridge.

    Returns
    -------
    np.ndarray, shape = (len(inds),2)
        array containing ridge points for current ridge.

    """

    if len(inds) >= min_ridge_pts:
        return ridge_pts[inds, :]


def ftle_ridges(f, eigvec_max, x, y, sdd_thresh=0.0, percentile=0, min_ridge_pts=3):
    """
    From ridge points from _ftle_ridges, extract connected ridges where
    a connected ridge is defined by a collection of points having continuous
    neighbors that are ridge points.

    Parameters
    ----------
    f : np.ndarray, shape = (nx,ny)
        ftle array.
    eigvec_max : np.ndarray, shape = (nx,ny,2)
        maximum eigenvector of Cauchy Green tensor.
    x : np.ndarray, shape = (nx,)
        array containing x-values.
    y : np.ndarray, shape = (ny,)
        array containing y-values.
    sdd_thresh : float, optional
        threshold for second directional derivative, should be at least 0.
        The default is 0.
    percentile : int, optional
        percentile of ftle used for min allowed value. The default is 0.
    min_ridge_pts : int
        minimum points allowed in ridge. The default is 3.

    Returns
    -------
    list
        list containing each connected ridge.

    """

    r_pts_, ridge_bool_, nx, ny = _ftle_ridges(
        f, eigvec_max, x, y, sdd_thresh=sdd_thresh, percentile=percentile
    )
    nx, ny = f.shape
    s = generate_binary_structure(2, 2)
    labels, nlabels = label(ridge_bool_, structure=s)
    ind_arr = np.arange(0, nx * ny).reshape(nx, ny)
    ridges = []
    for i in range(1, nlabels + 1):
        inds = ind_arr[labels == i]
        ridges.append(_get_ridges(r_pts_, inds, nx, ny, min_ridge_pts))

    return [r for r in ridges if r is not None]


@njit(parallel=True)
def _ftle_ridge_pts_connect(f, eigvec_max, x, y, sdd_thresh=0.0, percentile=0):
    """
    Compute FTLE ridge points by finding points (with subpixel accuracy) at which:
    ftle > 0 (or percentile of ftle),
    directional derivaitve of ftle (in eigvec_max direction) is 0,
    second directional derivative of ftle (in eigvec_max direction) is greater than
    sdd_thresh (in magnitude).

    Returns vectors at ridge points and second directional derivative value to be used
    later to connect ridge points and make curves.

    Parameters
    ----------
    f : np.ndarray, shape = (nx,ny)
        ftle array.
    eigvec_max : np.ndarray, shape = (nx,ny,2)
        maximum eigenvector of Cauchy Green tensor.
    x : np.ndarray, shape = (nx,)
        array containing x-values.
    y : np.ndarray, shape = (ny,)
        array containing y-values.
    sdd_thresh : float, optional
        threshold for second directional derivative, should be at least 0.
        The default is 0.
    percentile : int, optional
        percentile of ftle used for min allowed value. The default is 0.

    Returns
    -------
    r_pts : np.ndarray, shape = (k,3)
        ridge points and placeholder for ridge number.
    r_vec : np.ndarray, shape = (k,2)
        maximum eigenvector at grid point associated with ridge point.
    sdd : np.ndarray, shape = (k,)
        value of second directional derivative (in eigvec_max direction).
    h : float
        minimum grid spacing.

    """

    nx, ny = f.shape
    r_pts = -1 * np.ones((nx * ny, 3), numba.float64)
    r_vec = np.zeros((nx * ny, 2), numba.float64)
    sdd = np.zeros((nx * ny,), numba.float64)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    h = min(dx, dy)
    # set min value allowed for f
    if percentile == 0:
        f_min = 0
    else:
        f_min = np.percentile(f, percentile)

    for i in prange(2, nx - 2):
        for j in range(2, ny - 2):
            pt = np.array([x[i], y[j]])
            f0 = f[i, j]
            # compute derivatives if eigval large enough
            if f0 > f_min:
                fx = (f[i + 1, j] - f[i - 1, j]) / (2 * dx)
                fy = (f[i, j + 1] - f[i, j - 1]) / (2 * dy)
                fxx = (f[i + 1, j] - 2 * f[i, j] + f[i - 1, j]) / (dx**2)
                fyy = (f[i, j + 1] - 2 * f[i, j] + f[i, j - 1]) / (dy**2)
                fxy = (f[i + 1, j + 1] - f[i + 1, j - 1] - f[i - 1, j + 1] + f[i - 1, j - 1]) / (
                    4 * dx * dy
                )

                eigvec_max0 = eigvec_max[i, j, :]
                ex, ey = eigvec_max0

                # compute second directional derivative in eigvec direction
                c2 = ex * (fxx * ex + fxy * ey) + ey * (fxy * ex + fyy * ey)

                # if second directional derivative is negative and large enough,
                # use taylor expansion of f to find subpixel ridge point
                if c2 < -sdd_thresh:
                    t = -(fx * ex + fy * ey) / c2
                    # if point lies within current grid, keep point and data
                    if abs(t * ex) <= dx / 2 and abs(t * ey) <= dy / 2:
                        k = i * ny + j
                        r_pts[k, :2] = pt + np.array([t * ex, t * ey])
                        r_vec[k, :] = eigvec_max0
                        sdd[k] = c2

    return r_pts, r_vec, sdd, h


@njit
def _link_points_stepper(
    pt0, nvec0, ind0, r_pts, r_vecs, sdd, offset_ravel_inds, ridge_num, a, c=1.0, sdd_thresh=0.0
):
    """
    Searches for nearby ridge point in direction tangent to ridge,
    if none are found, return ridge points and none for neighbor.

    Parameters
    ----------
    pt0 : np.ndarray, shape = (2,)
        current point.
    nvec0 : np.ndarray, shape = (2,)
        current normal vector to ridge at pt.
    ind0 : int
        index corresponding to where pt is in r_pts.
    r_pts : np.ndarray, shape = (nrpts,3)
        array containing ridge points and what ridge the point belongs to.
    r_vecs : np.ndarray, shape = (nrpts,2)
        array containing normal vector to each point in r_pts.
    sdd : np.ndarray, shape = (nrpts,)
        value of second directional derivative for each point in r_pts.
    offset_ravel_inds : np.ndarray, shape = (10,)
        indices corresponding to directional offset from a meshgrid but in raveled form.
    ridge_num : int
        current ridge number.
    a : float
        scaling for distance portion of metric.
    c : float, optional
        scaling for angle portion of metric. The default is 1..
    sdd_thresh : float, optional
        threshold for second directional derivative, should be at least 0.
        The default is 0.

    Returns
    -------
    r_pts : np.ndarray, shape = (nrpts,3)
        array containing ridge points and what ridge the point belongs to with
        index neighbor of neighbor updated if found.
    cand_pt : None or np.ndarray, shape = (2,)
        linked ridge point if found, else None is returned.
    pt_vec : None or np.ndarray, shape = (2,)
        normal vec corresponding to linked point if found, else None is returned.
    pt_ind : None or int
        index corresponding to linked point if found, else None is returned.

    """

    # convert angle into index corresponding to search direction
    angle_ind = floor((atan2(nvec0[0], -nvec0[1]) % (2 * pi)) * 4 / pi + 0.5) % 8
    offset_inds = offset_ravel_inds[angle_ind : angle_ind + 3]
    cand_pts = np.zeros((3, 2), numba.float64)
    metric = np.zeros((3), numba.float64)
    pt_vecs = np.zeros((3, 2), numba.float64)

    # loop through candidate neighbors
    nhbrs_flag = False
    k_off = np.zeros((3,), numba.int32)

    for i in range(3):
        k_off_ = ind0 + offset_inds[i]
        k_off[i] = k_off_
        cr_pts = r_pts[k_off_, :]
        cand_pts[i, :] = cr_pts[:2]
        # if neighbor is ridge point, compute linking metric
        if sdd[k_off_] + sdd_thresh < 0:
            d = ((cand_pts[i, 0] - pt0[0]) ** 2 + (cand_pts[i, 1] - pt0[1]) ** 2) ** 0.5
            nvec1 = np.array([r_vecs[k_off_, 0], r_vecs[k_off_, 1]])

            dot01 = nvec0[0] * nvec1[0] + nvec0[1] * nvec1[1]

            if dot01 < 0:
                pt_vecs[i, :] = -nvec1
                dot01 = -dot01
            else:
                pt_vecs[i, :] = nvec1

            beta = acos(dot01)
            metric[i] = a * d + c * beta
            nhbrs_flag = True
        else:
            metric[i] = 10000.0

    if nhbrs_flag:
        nhbr_ind = np.argmin(metric)
        pt_ind = k_off[nhbr_ind]
        if r_pts[pt_ind, 2] + 0.5 < 0:
            r_pts[pt_ind, 2] = ridge_num
            return r_pts, cand_pts[nhbr_ind, :], pt_vecs[nhbr_ind, :], pt_ind

        else:
            return r_pts, None, None, None

    else:
        return r_pts, None, None, None


@njit
def _linked_ridge_pts(f, eigvec_max, x, y, sdd_thresh=0.0, percentile=0, c=1.0):
    """
    Goes through all ridge points, orders and groups them into ridges,
    if a point has a neighbor in the eigvec_max direction (meant to approximate
    the tangent of the ridge), points are connected and added to ridge.

    Parameters
    ----------
    f : np.ndarray, shape = (nx,ny)
        ftle array.
    eigvec_max : np.ndarray, shape = (nx,ny,2)
        maximum eigenvector of Cauchy Green tensor.
    x : np.ndarray, shape = (nx,)
        array containing x-values.
    y : np.ndarray, shape = (ny,)
        array containing y-values.
    sdd_thresh : float, optional
        threshold for second directional derivative, should be at least 0.
        The default is 0.
    percentile : int, optional
        percentile of ftle used for min allowed value. The default is 0.
    c : float, optional
        scaling for angle portion of metric. The default is 1..

    Returns
    -------
    linked_ridges_arr : np.ndarray, shape = (ridge_pt_counter,2)
        array containing ordered ridge point grouped by ridge number
    ridge_len : np.ndarray, shape = (ridge_num_counter,2)
        array containing ending indices for each ridge and
        its corresponding length.
    endpoints : np.ndarray, shape = (2*ridge_num_counter,3)
        array containing endpoints for each ridge, final column is ridge number
        where k.0 is the begining of ridge and -k.1 is the end.
    ep_tanvecs : np.ndarray, shape = (2*ridge_num_counter,2)
        array containing properly oriented tangent vectors corresponding to
        each endpoint.

    """

    # compute ridge points and store normal vectors and 2nd directional derivative
    # value corresponding to each point
    rpts, rvecs, sdd, h = _ftle_ridge_pts_connect(
        f, eigvec_max, x, y, sdd_thresh=sdd_thresh, percentile=percentile
    )

    a = 1 / h
    # get indices corresponding to pts sorted by 2nd directional derivative value
    sorted_inds = np.argsort(sdd)

    # create raveled version of grid offset inds
    nx, ny = f.shape
    grid_shape = np.array([nx, ny], np.int32)
    offset_inds_arr = np.array(
        [[1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]]
    )
    offset_ravel_inds = np.zeros(10, np.int32)
    for i, inds in enumerate(offset_inds_arr):
        offset_ravel_inds[i] = ravel_index(inds, grid_shape)

    # sdd_max = -sdd[sorted_inds[0]]
    # up_thresh = sdd_thresh + 0.01*sdd_max
    up_thresh = sdd_thresh
    ridge_num_counter = 0  # counts which ridge
    linked_ridges_arr = np.zeros((nx * ny, 2), np.float64)
    ridge_len = np.zeros((nx * ny, 2), np.int32)
    endpoints = np.zeros((nx * ny, 3), np.float64)
    ep_tanvecs = np.zeros((nx * ny, 2), np.float64)
    ridge_pt_counter = 0  # counts how many points
    ridge0_start = 0  # keeps track of index current ridge started at
    # loop through points is descending value (in magnitude) of 2nd directional derivative
    for si in sorted_inds:
        ridge_num = rpts[si, -1]
        # check if current points alreadly belongs to a ridge, if yes, skip
        if ridge_num + 0.5 < 0:
            # check if 2nd directional derivative is above starting threshold
            if sdd[si] < -up_thresh:
                rpts[si, -1] = ridge_num_counter  # set points ridge number to current
                pt0 = rpts[si, :2].copy()  # set starting point, vec, ind
                nvec0 = rvecs[si, :].copy()
                ind0 = si

                # set first point in ridge array
                linked_ridges_arr[ridge_pt_counter, :] = np.array([pt0[0], pt0[1]], np.float64)

                ridge_pt_counter += 1
                # do I need .copy() here and below?

                # set endpoint
                ept0 = pt0.copy()

                # find neighbors
                for k in range(10000):
                    rpts, pt0, nvec0, ind0 = _link_points_stepper(
                        pt0,
                        nvec0,
                        ind0,
                        rpts,
                        rvecs,
                        sdd,
                        offset_ravel_inds,
                        ridge_num_counter,
                        a,
                        c=c,
                    )
                    # if no more neighbors, break
                    if ind0 is None:
                        break

                    # set endpoint
                    ept0 = pt0.copy()

                    # add point to array
                    linked_ridges_arr[ridge_pt_counter, :] = np.array([pt0[0], pt0[1]], np.float64)
                    ridge_pt_counter += 1

                # flip ridge so ridge is in order
                linked_ridges_arr[ridge0_start:ridge_pt_counter, :] = np.flipud(
                    linked_ridges_arr[ridge0_start:ridge_pt_counter, :]
                )
                # same process as above but in the other direction
                pt1 = rpts[si, :2].copy()
                ind1 = si
                nvec1 = -rvecs[si, :].copy()
                ept1 = pt1.copy()
                for kk in range(10000):
                    rpts, pt1, nvec1, ind1 = _link_points_stepper(
                        pt1,
                        nvec1,
                        ind1,
                        rpts,
                        rvecs,
                        sdd,
                        offset_ravel_inds,
                        ridge_num_counter,
                        a,
                        c=c,
                    )

                    if ind1 is None:
                        break
                    ept1 = pt1.copy()
                    linked_ridges_arr[ridge_pt_counter, :] = np.array([pt1[0], pt1[1]], np.float64)
                    ridge_pt_counter += 1

                # if ridge only consists of one point, remove from array and reset counter
                # to value on previous iterate
                if ridge0_start == ridge_pt_counter - 1:
                    linked_ridges_arr[ridge0_start, :] = np.array([0.0, 0.0])
                    ridge_pt_counter -= 1
                    ridge0_start = ridge_pt_counter
                # else, save endpoints, endpoint vecs, and update start ind and ridge num
                else:
                    epvec0 = linked_ridges_arr[ridge0_start + 1, :] - ept0
                    epvec0 = epvec0 / (epvec0[0] ** 2 + epvec0[1] ** 2) ** 0.5
                    epvec1 = ept1 - linked_ridges_arr[ridge_pt_counter - 2, :]
                    epvec1 = epvec1 / (epvec1[0] ** 2 + epvec1[1] ** 2) ** 0.5
                    ridge_len[ridge_num_counter, :] = np.array(
                        [ridge_pt_counter, ridge_pt_counter - ridge0_start]
                    )

                    endpoints[2 * ridge_num_counter : 2 * ridge_num_counter + 2, :] = np.array(
                        [
                            [ept0[0], ept0[1], ridge_num_counter],
                            [ept1[0], ept1[1], -(ridge_num_counter + 0.1)],
                        ]
                    )
                    ep_tanvecs[2 * ridge_num_counter : 2 * ridge_num_counter + 2, :] = np.array(
                        [[epvec0[0], epvec0[1]], [epvec1[0], epvec1[1]]]
                    )
                    ridge0_start = ridge_pt_counter
                    ridge_num_counter += 1

            else:
                break

        else:
            continue

    return (
        linked_ridges_arr[:ridge_pt_counter, :],
        ridge_len[:ridge_num_counter, :],
        endpoints[: 2 * ridge_num_counter, :],
        ep_tanvecs[: 2 * ridge_num_counter, :],
    )


@njit
def _endpoint_distances(pt, arr, dist_tol):
    """
    Computes distances between pt and all other points in arr,
    tangent vectors between pt and all other points are computed
    if point from arr is within dist_tol of pt.

    Parameters
    ----------
    pt : np.ndarray, shape = (2,)
        current point.
    arr : np.ndarray, shape = (npts,2)
        points which distance from pt are computed.
    dist_tol : float
        tolerance used for distance.

    Returns
    -------
    dist : np.ndarry, shape = (npts,)
        array containing distances.
    tan_vec : np.ndarray, shape = (npts,2)
        array containing tangent vectors.
    tol_bool : np.ndarray, shape = (npts,)
        truth values which deterimines if kth point is within dist_tol of pt.

    """

    len_arr = arr.shape[0]
    dist = np.zeros(len_arr, numba.float64)
    tan_vec = np.zeros_like(arr, numba.float64)
    tol_bool = np.zeros(len_arr, numba.bool_)
    for k in prange(len_arr):
        tan_vec[k, :] = np.array([arr[k, 0] - pt[0], arr[k, 1] - pt[1]])
        dist[k] = (tan_vec[k, 0] ** 2 + tan_vec[k, 1] ** 2) ** 0.5
        if dist[k] < dist_tol:
            tol_bool[k] = 1
            tan_vec[k, :] = tan_vec[k, :] / (tan_vec[k, 0] ** 2 + tan_vec[k, 1] ** 2) ** 0.5
    return dist, tan_vec, tol_bool


# @njit
def _connect_endpoints(
    endpoint, current_tan_vec, rem_endpoints, rem_endpoints_tan, ep_tan_ang, dist_tol
):
    """
    Finds closest point from rem_endpoints to endpoint such that the angle between
    ridge tangent at each point and the tangent between the points is less than ep_tan_ang,
    removes point and other endpoint of same ridge from rem_endpoints if conditions are
    satisfied. Returns rem_endpoint arrays and None for new_endpoint if none is found.

    Parameters
    ----------
    endpoint : np.ndarray, shape = (2,)
        current endpoint.
    current_tan_vec : np.ndarray, shape = (2,)
        tangent vector at current endpoint.
    rem_endpoints : np.ndarray, shape = (nrem,3)
        array containing remaining endpoints and their ridge numbers.
    rem_endpoints_tan : np.ndarray, shape = (nrem,2)
        array containing tangent vector at remaining endpoints.
    ep_tan_ang : float
        angle used to confirm endpoints are in line.
    dist_tol : float
        tolerance used to consider new endpoint for connection.

    Returns
    -------
    rem_endpoints : np.ndarray, either shape = (nrem,3) or shape = (nrem-2,3)
        array containing remaining endpoints and their ridge numbers.
    rem_endpoints_tan : np.ndarray, either shape = (nrem,2) or shape = (nrem-2,2)
        array containing tangent vector at remaining endpoints.
    connect_ind : int
        index corresponding to endpoint to connect if found, else return -1.
    new_endpoint : None or np.ndarray, shape = (2,)
        new_endpoint to use in search if found, else return None.

    """

    dist_ep, tan_ep, dist_mask = _endpoint_distances(endpoint[:-1], rem_endpoints[:, :-1], dist_tol)

    keep_mask = np.ones(rem_endpoints.shape[0], np.bool_)
    connect_ind = -1
    new_endpoint = None
    if dist_mask.any():
        ep_sign = int(copysign(1, endpoint[-1]))
        n_in_tol = dist_mask.sum()
        for k in range(n_in_tol):
            ep_ind = np.argmin(dist_ep)
            new_ep_sign = int(copysign(1, rem_endpoints[ep_ind, -1]))
            tan_ep0 = tan_ep[ep_ind, :]
            new_tan_vec = rem_endpoints_tan[ep_ind, :]

            if (
                acos(-ep_sign * (tan_ep0[0] * current_tan_vec[0] + tan_ep0[1] * current_tan_vec[1]))
                < ep_tan_ang
                and acos(new_ep_sign * (new_tan_vec[0] * tan_ep0[0] + new_tan_vec[1] * tan_ep0[1]))
                < ep_tan_ang
            ):
                new_ridge_ind = rem_endpoints[ep_ind, -1]
                ep_sign = int(copysign(1, new_ridge_ind))
                connect_ind = new_ridge_ind
                new_endpoint = rem_endpoints[ep_ind + ep_sign, :]
                current_tan_vec = rem_endpoints_tan[ep_ind + ep_sign, :]
                keep_mask[ep_ind] = False
                keep_mask[ep_ind + ep_sign] = False
                rem_endpoints = rem_endpoints[keep_mask, :]
                rem_endpoints_tan = rem_endpoints_tan[keep_mask, :]
                break
            else:
                dist_ep[ep_ind] = 10 * dist_tol

    return rem_endpoints, rem_endpoints_tan, connect_ind, new_endpoint, current_tan_vec


def ftle_ordered_ridges(
    f,
    eigvec_max,
    x,
    y,
    dist_tol,
    ep_tan_ang=pi / 4,
    min_ridge_pts=5,
    sdd_thresh=0.0,
    percentile=0,
    c=1.0,
):
    """
    Computes ftle ridge points and links points into ridges. After
    this is done, ridges that should be connected are searched for
    and connected if they meet certain critera. Criteria: if two ridges have
    endpoints that are within dist_tol and the angle between the tangent of
    the ridge at the endpoints and vector connecting the endpoints is less
    than ep_tan_ang, connect ridges.

    Parameters
    ----------
    f : np.ndarray, shape = (nx,ny)
        ftle array.
    eigvec_max : np.ndarray, shape = (nx,ny,2)
        maximum eigenvector of Cauchy Green tensor.
    x : np.ndarray, shape = (nx,)
        array containing x-values.
    y : np.ndarray, shape = (ny,)
        array containing y-values.
    dist_tol : float
        tolerance used to consider new endpoint for connection.
    ep_tan_ang : float
        angle used to confirm endpoints are in line. The default is pi/4.
    min_ridge_pts : int, optional
        minimum points allowed in ridge after endpoints are connected. The default is 5.
    sdd_thresh : float, optional
        threshold for second directional derivative, should be at least 0.0.
        The default is 0.
    percentile : int, optional
        percentile of ftle used for min allowed value. The default is 0.
    c : float, optional
        scaling for angle portion of metric used in linking algorithm. The default is 1.0.

    Returns
    -------
    ridges : list
        list containing connected, ordered ridges.

    """

    ridge_pts, ridge_lens, endpoints, ep_tanvecs = _linked_ridge_pts(
        f, eigvec_max, x, y, sdd_thresh=sdd_thresh, percentile=percentile, c=c
    )

    rem_endpoints = endpoints.copy()
    rem_endpoints_tan = ep_tanvecs.copy()
    nridges = int(endpoints.shape[0] / 2)
    ridges = []

    # while endpoints remain, search for nearby endpoints
    while rem_endpoints.size > 0:
        # set current endpoints and tan vectors
        current_endpoints = rem_endpoints[:2, :]
        current_tan_vecs = rem_endpoints_tan[:2, :]
        mask = np.ones(rem_endpoints.shape[0], np.bool_)

        # remove current from remaining
        mask[:2] = False

        rem_endpoints = rem_endpoints[mask, :]
        rem_endpoints_tan = rem_endpoints_tan[mask, :]

        # compute endpoint distances for current endpoints (both sides)
        dist_ep = np.zeros((2, rem_endpoints.shape[0]), np.float64)
        dist_mask = np.zeros((2, rem_endpoints.shape[0]), np.bool_)

        # maybe remove tan_ep
        for i in range(2):
            dist_ep[i, :], _, dist_mask[i, :] = _endpoint_distances(
                current_endpoints[i, :-1], rem_endpoints[:, :-1], dist_tol
            )

        # tells if other endpoints within dist for both current endpoints
        ep_in_tol = dist_mask[0, :].any() + 2 * dist_mask[1, :].any()

        # if both endpoints have other endpoints in dist
        if ep_in_tol == 3:
            connect_inds0 = np.zeros((nridges,), np.float64)
            ii = 0
            # find which has closer endpoint
            min0 = min(dist_ep[0, :])
            min1 = min(dist_ep[1, :])
            ep_ind = np.argmin(np.array([min0, min1]))
            # add current endpoint to connect_inds0 and increment
            ridge_ep_ind = current_endpoints[ep_ind, -1]
            connect_inds0[ii] = -(ridge_ep_ind + 0.1)
            ii += 1
            new_endpoint = current_endpoints[ep_ind, :]
            current_tan_vec = current_tan_vecs[ep_ind, :]
            # keep searching for endpoints for current ridge until there are none
            for kk in range(nridges):
                (rem_endpoints, rem_endpoints_tan, connect_ind, new_endpoint, current_tan_vec) = (
                    _connect_endpoints(
                        new_endpoint,
                        current_tan_vec,
                        rem_endpoints,
                        rem_endpoints_tan,
                        ep_tan_ang,
                        dist_tol,
                    )
                )
                # break if no more endpoints
                if new_endpoint is None:
                    break

                # add index of endpoints to current ridge
                connect_inds0[ii] = connect_ind
                ii += 1

            # do same thing for other endpoint
            connect_inds1 = np.zeros((nridges,), np.float64)
            jj = 0
            ep_sign = int(copysign(1, ridge_ep_ind))
            ep_ind += ep_sign
            # add current endpoint to connect_inds and increment
            ridge_ep_ind = current_endpoints[ep_ind, -1]
            connect_inds1[jj] = -(ridge_ep_ind + 0.1)
            jj += 1
            new_endpoint = current_endpoints[ep_ind, :]
            current_tan_vec = current_tan_vecs[ep_ind, :]
            # keep searching for endpoints for current ridge until there are none
            for kk in range(nridges):
                (rem_endpoints, rem_endpoints_tan, connect_ind, new_endpoint, current_tan_vec) = (
                    _connect_endpoints(
                        new_endpoint,
                        current_tan_vec,
                        rem_endpoints,
                        rem_endpoints_tan,
                        ep_tan_ang,
                        dist_tol,
                    )
                )
                # break if no more endpoints
                if new_endpoint is None:
                    break

                # add index of endpoints to current ridge
                connect_inds1[jj] = connect_ind
                jj += 1

            # if endpoints are found, add to connect_inds array
            if ii > 1 and jj > 1:
                connect_inds = np.hstack(
                    (np.flipud(-(connect_inds0[:ii] + 0.1)), connect_inds1[1:jj])
                )
            elif ii > 1:
                connect_inds = connect_inds0[:ii]
            else:
                connect_inds = connect_inds1[:jj]
            inds = np.round(np.abs(connect_inds)).astype(np.int32)

            if ii + jj > 2:
                rinds = ridge_lens[inds, 0]
                rlens = ridge_lens[inds, 1]
                full_len = rlens.sum()
                if full_len < min_ridge_pts:
                    continue
                ridge_arr = np.zeros((full_len, 2), np.float64)
                if connect_inds[0] + 0.01 < 0:
                    ridge_arr[0 : rlens[0], :] = np.flipud(
                        ridge_pts[rinds[0] - rlens[0] : rinds[0], :]
                    )
                else:
                    ridge_arr[0 : rlens[0], :] = ridge_pts[rinds[0] - rlens[0] : rinds[0], :]
                j = rlens[0]
                for i, ri in enumerate(connect_inds[1:]):
                    if ri + 0.01 < 0:
                        ridge_arr[j : j + rlens[i + 1], :] = np.flipud(
                            ridge_pts[rinds[i + 1] - rlens[i + 1] : rinds[i + 1], :]
                        )

                    else:
                        ridge_arr[j : j + rlens[i + 1], :] = ridge_pts[
                            rinds[i + 1] - rlens[i + 1] : rinds[i + 1], :
                        ]

                    j += rlens[i + 1]

                ridges.append(ridge_arr)
            else:
                rind = ridge_lens[inds, 0][0]
                rlen = ridge_lens[inds, 1][0]
                if rlen < min_ridge_pts:
                    continue
                ridges.append(ridge_pts[rind - rlen : rind, :])

        # if only the ending endpoint
        elif ep_in_tol == 2:
            ii = 0
            ep_ind = 1
            ridge_ep_ind = current_endpoints[ep_ind, -1]
            connect_inds = np.zeros((nridges,), np.float64)
            connect_inds[ii] = -(ridge_ep_ind + 0.1)
            ii += 1
            new_endpoint = current_endpoints[ep_ind, :]
            current_tan_vec = current_tan_vecs[ep_ind, :]
            # keep searching for endpoints for current ridge until there are none
            for kk in range(nridges):
                (rem_endpoints, rem_endpoints_tan, connect_ind, new_endpoint, current_tan_vec) = (
                    _connect_endpoints(
                        new_endpoint,
                        current_tan_vec,
                        rem_endpoints,
                        rem_endpoints_tan,
                        ep_tan_ang,
                        dist_tol,
                    )
                )
                # break if no more endpoints
                if new_endpoint is None:
                    break

                # add index of endpoints to current ridge
                connect_inds[ii] = connect_ind
                ii += 1
            connect_inds = connect_inds[:ii]
            inds = np.round(np.abs(connect_inds)).astype(np.int32)
            if ii > 1:
                rinds = ridge_lens[inds, 0]
                rlens = ridge_lens[inds, 1]
                full_len = rlens.sum()
                if full_len < min_ridge_pts:
                    continue
                ridge_arr = np.zeros((full_len, 2), np.float64)
                if connect_inds[0] + 0.01 < 0:
                    ridge_arr[0 : rlens[0], :] = np.flipud(
                        ridge_pts[rinds[0] - rlens[0] : rinds[0], :]
                    )
                else:
                    ridge_arr[0 : rlens[0], :] = ridge_pts[rinds[0] - rlens[0] : rinds[0], :]
                j = rlens[0]
                for i, ri in enumerate(connect_inds[1:]):
                    if ri + 0.01 < 0:
                        ridge_arr[j : j + rlens[i + 1], :] = np.flipud(
                            ridge_pts[rinds[i + 1] - rlens[i + 1] : rinds[i + 1], :]
                        )

                    else:
                        ridge_arr[j : j + rlens[i + 1], :] = ridge_pts[
                            rinds[i + 1] - rlens[i + 1] : rinds[i + 1], :
                        ]

                    j += rlens[i + 1]

                ridges.append(ridge_arr)
            else:
                rind = ridge_lens[inds, 0][0]
                rlen = ridge_lens[inds, 1][0]
                if rlen < min_ridge_pts:
                    continue
                ridges.append(ridge_pts[rind - rlen : rind, :])

        # if only the starting endpoint
        elif ep_in_tol == 1:
            ii = 0
            ep_ind = 0
            ridge_ep_ind = current_endpoints[ep_ind, -1]
            connect_inds = np.zeros((nridges,), np.float64)
            connect_inds[ii] = -(ridge_ep_ind + 0.1)
            ii += 1
            new_endpoint = current_endpoints[ep_ind, :]
            current_tan_vec = current_tan_vecs[ep_ind, :]
            # keep searching for endpoints for current ridge until there are none
            for kk in range(nridges):
                (rem_endpoints, rem_endpoints_tan, connect_ind, new_endpoint, current_tan_vec) = (
                    _connect_endpoints(
                        new_endpoint,
                        current_tan_vec,
                        rem_endpoints,
                        rem_endpoints_tan,
                        ep_tan_ang,
                        dist_tol,
                    )
                )
                # break if no more endpoints
                if new_endpoint is None:
                    break

                # add index of endpoints to current ridge
                connect_inds[ii] = connect_ind
                ii += 1
            connect_inds = connect_inds[:ii]
            inds = np.round(np.abs(connect_inds)).astype(np.int32)
            if ii > 1:
                rinds = ridge_lens[inds, 0]
                rlens = ridge_lens[inds, 1]
                full_len = rlens.sum()
                if full_len < min_ridge_pts:
                    continue
                ridge_arr = np.zeros((full_len, 2), np.float64)
                if connect_inds[0] + 0.01 < 0:
                    ridge_arr[0 : rlens[0], :] = np.flipud(
                        ridge_pts[rinds[0] - rlens[0] : rinds[0], :]
                    )
                else:
                    ridge_arr[0 : rlens[0], :] = ridge_pts[rinds[0] - rlens[0] : rinds[0], :]
                j = rlens[0]
                for i, ri in enumerate(connect_inds[1:]):
                    if ri + 0.01 < 0:
                        ridge_arr[j : j + rlens[i + 1], :] = np.flipud(
                            ridge_pts[rinds[i + 1] - rlens[i + 1] : rinds[i + 1], :]
                        )

                    else:
                        ridge_arr[j : j + rlens[i + 1], :] = ridge_pts[
                            rinds[i + 1] - rlens[i + 1] : rinds[i + 1], :
                        ]

                    j += rlens[i + 1]

                ridges.append(ridge_arr)
            else:
                rind = ridge_lens[inds, 0][0]
                rlen = ridge_lens[inds, 1][0]
                ridges.append(ridge_pts[rind - rlen : rind, :])
        else:
            inds = int(current_endpoints[0, -1])
            rind = ridge_lens[inds, 0]
            rlen = ridge_lens[inds, 1]
            if rlen < min_ridge_pts:
                continue
            ridges.append(ridge_pts[rind - rlen : rind, :])

    return ridges

import numpy as np
from numba import njit, prange
import numba
from math import floor, pi, cos, sin
from scipy.interpolate import splprep, splev

@njit
def unravel_index(index,shape):
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
    arr_ind = np.zeros(len(shape),numba.int64)
    ind = index
    for i in range(len(shape)-1):
        div_mod = divmod(ind,shape[i])
        arr_ind[i] = div_mod[1]
        ind = div_mod[0]
        
    arr_ind[-1] = div_mod[0]
    
    return np.flip(arr_ind)


@njit
def ravel_index(inds,shape):
    """
    Finds raveled index corresponding to grid index given by inds from array with
    shape=shape where shape must be a numpy.array 

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
    for i,ind in enumerate(inds[:-1]):
        r_ind+=ind*np.prod(shape[i+1:])
        
    return r_ind

            
@njit
def finite_diff_2D(f,i,j,h,axis,direction='c'):
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

    if axis==0:
        if direction == 'c':
            df = (f[i+1,j] - f[i-1,j])/(2*h)
        elif direction == 'b':
            df = (3*f[i,j] - 4*f[i-1,j] + f[i-2,j])/(2*h)
        elif direction == 'f':
            df = (-f[i+2,j] + 4*f[i+1,j] - 3*f[i,j])/(2*h)
        else:
            print("Valid difference directions are 'c', 'b', and 'f'" )
    if axis==1:
        if direction == 'c':
            df = (f[i,j+1] - f[i,j-1])/(2*h)
        elif direction == 'b':
            df = (3*f[i,j] - 4*f[i,j-1] + f[i,j-2])/(2*h)
        elif direction == 'f':
            df = (-f[i,j+2] + 4*f[i,j+1] - 3*f[i,j])/(2*h)
        else:
            print("Valid difference directions are 'c', 'b', and 'f'" )  
    
    return df


@njit
def finite_diff_ND(f,ind,h,axis,shape,direction=0):
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

    di = np.zeros(len(ind),numba.int64)
    di[axis] = 1
    if direction == 0:
        i_p1 = ravel_index(ind+di,shape)
        i_m1 = ravel_index(ind-di,shape)
        df = (f[i_p1] - f[i_m1])/(2*h)
    elif direction == -1:
        i = ravel_index(ind,shape)
        i_m1 = ravel_index(ind-di,shape)
        i_m2 = ravel_index(ind-2*di,shape)
        df = (3*f[i] - 4*f[i_m1] + f[i_m2])/(2*h)
    elif direction == 1:
        i = ravel_index(ind,shape)
        i_p1 = ravel_index(ind+di,shape)
        i_p2 = ravel_index(ind+2*di,shape)
        df = (-f[i_p2] + 4*f[i_p1] - 3*f[i])/(2*h)
    else:
        print("Valid difference directions are 0: centered, -1: backward, and 1: forward" )
    
    return df


@njit
def finite_diff_2D_2nd(f,i,j,h,axis,direction='c'):
    """
    Compute 2nd order partial finite difference in the array f @ [i,j] along axis=axis and using a
    directional difference scheme defined by direction for the second derivative. axis=2 is for the
    mixed partial finite difference
    """
    if axis==0:
        if direction == 'c':
            df = (f[i+1,j] - 2*f[i,j] + f[i-1,j])/(h**2)
        elif direction == 'b':
            df = (2*f[i,j] - 5*f[i-1,j] + 4*f[i-2,j] - f[i-3,j])/(h**3)
        elif direction == 'f':
            df = (2*f[i,j] - 5*f[i+1,j] + 4*f[i+2,j] - f[i+3,j])/(h**3)
        else:
            print("Valid difference directions are 'c', 'b', and 'f'" )
    if axis==1:
        if direction == 'c':
            df = (f[i,j+1] - 2*f[i,j] + f[i,j-1])/(h**2)
        elif direction == 'b':
            df = (2*f[i,j] - 5*f[i,j-1] + 4*f[i,j-2] - f[i,j-3])/(h**3)
        elif direction == 'f':
            df = (2*f[i,j] - 5*f[i,j+1] + 4*f[i,j+2] - f[i,j+3])/(h**3)
        else:
            print("Valid difference directions are 'c', 'b', and 'f'" )  
    if axis==2:
        if direction == 'c':
            df = (f[i+1,j+1] - f[i+1,j-1] - f[i-1,j+1] + f[i-1,j-1])/(4*h**2)
        else:
            print("Valid difference directions are 'c' for mixed derivative" ) 
    return df



@njit(parallel=True)
def curl_vel(u,v,dx,dy):
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
    nx,ny = u.shape
    curl = np.zeros((nx,ny),numba.float64)
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            dfydx = (v[i+1,j] - v[i-1,j])/(2*dx)
            dfxdy = (u[i,j+1] - u[i,j-1])/(2*dy)
            
            curl[i,j] = dfydx - dfxdy
    
    return curl


@njit(parallel=True)
def curl_vel_tspan(u,v,dx,dy):
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
    nt,nx,ny = u.shape
    curl = np.zeros((nt,nx,ny),numba.float64)
    for k in prange(nt):
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                dfydx = (v[k,i+1,j] - v[k,i-1,j])/(2*dx)
                dfxdy = (u[k,i,j+1] - u[k,i,j-1])/(2*dy)
                
                curl[k,i,j] = dfydx - dfxdy
    
    return curl
    
    
@njit(parallel=True)
def curl_func(fnc,x,y,h=1e-3):
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
    curlf = np.zeros((nx,ny),numba.float64)
    dx_vec = np.array([h,0.0],numba.float64)
    dy_vec = np.array([0.0,h],numba.float64)
    for i in prange(nx):
        for j in range(ny):
            pt = np.array([x[i],y[j]])
            
            dfydx = (fnc(pt+dx_vec)[1] - fnc(pt-dx_vec)[1])/(2*h)
            dfxdy = (fnc(pt+dy_vec)[0] - fnc(pt-dy_vec)[0])/(2*h)
            
            curlf[i,j] = dfydx - dfxdy
    
    return curlf


@njit(parallel=True)
def curl_func_tspan(fnc,t,x,y,h=1e-3):
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
    nt = len(t)
    nx = len(x)
    ny = len(y)
    curlf = np.zeros((nt,nx,ny),numba.float64)
    dx_vec = np.array([0.0,h,0.0],numba.float64)
    dy_vec = np.array([0.0,0.0,h],numba.float64)
    for k in prange(nt):
        for i in range(nx):
            for j in range(ny):
                pt = np.array([t[k],x[i],y[j]])
                
                dfydx = (fnc(pt+dx_vec)[1] - fnc(pt-dx_vec)[1])/(2*h)
                dfxdy = (fnc(pt+dy_vec)[0] - fnc(pt-dy_vec)[0])/(2*h)
                
                curlf[k,i,j] = dfydx - dfxdy
    
    return curlf 


@njit
def composite_simpsons_38(f,h):
    """
    Composite Simpson's 3/8 rule to compute integral of f between endpoitns of pts with
    regular spacing given by h.

    Parameters
    ----------
    f : np.ndarray, shape = (n,)
        values of f at irregularly spaced points.
    h : float
        value of spacing between points at which f was evaluated.

    Returns
    -------
    val : float
        value of integral.

    """
    n = f.shape[0]
    val = 0
    for k in range(n):
        if k == 0 or k == n-1:
            val += f[k]
        elif k%3 != 0:
            val += 3*f[k]
        else:
            val += 2*f[k]
            
    return (3*h/8)*val


@njit
def composite_simpsons_38_irregular(f,h):
    """
    Composite Simpson's 3/8 rule to compute integral of f between endpoitns of pts with
    irregular spacing given by h which is an array.

    Parameters
    ----------
    f : np.ndarray, shape = (n,)
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
    for k in range(int(n/2)):
        h0 = h[2*k]
        h1 = h[2*k+1]
        val+=(1/6)*(h0+h1)*(f[2*k]*(2-h1/h0) + 
                            f[2*k+1]*((h0+h1)**2)/(h0*h1) + 
                            f[2*k+2]*(2-h0/h1))
    
    if n%2 == 1:
        h1 = h[n-1]
        h2 = h[n-2]
        alph = (2*h1**2 + 3*h1*h2)/(6*(h2 + h1))
        beta = (h1**2 + 3*h1*h2)/(6*h2)
        eta = (h1**3)/(6*h2*(h2 + h1))
        
        val+=alph*f[-1] + beta*f[-2] - eta*f[-3]
            
    return val

@njit
def dist_2d(p1,p2):
    """
    Compute 2D Euclidean distance between p1 and p2

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
    
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5


@njit
def dist_tol(point,arr,tol):
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
    k=0
    dist = np.zeros(len(arr),numba.float64)
    near = False
    for pt in arr:
        dist[k] = dist_2d(pt,point)
        if dist[k] < tol:
            near = True
            break
        k+=1
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
    x0 = polygon[0,0]
    y1 = polygon[1,1]
    area = x0*y1
    for k in range(1,n-1):
        x1 = polygon[k,0]
        y2 = polygon[k+1,1]
        y0 = polygon[k-1,1]
        area += x1*(y2 - y0)
        
    area = abs(0.5*(area - polygon[n-1,0]*polygon[n-2,1]))
                
    return area



@njit
def max_in_radius(arr,r,dx,dy,n=-1,min_val=0.0):
    """
    Finds n local maxima values in arr such that each max is a local maximum within radius r where
    spacing in arr is given by dx,dy. If all local maxima are desired, set n = -1. Should pass a
    copy of arr to the function to avoid orginial arr being overwritten (i.e. pass in arr.copy())

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

    nx,ny = arr.shape
    ix = floor(r/dx)
    iy = floor(r/dy)
    arr_shape = np.array([nx,ny])
    if n == -1:
        max_inds = np.zeros((int(nx*ny/2),2),numba.int32)
        max_vals = np.zeros(int(nx*ny/2),numba.float64)
        k=0
        while np.max(arr) > min_val:
            max_ind = np.argmax(arr)
            max_inds[k,:] = unravel_index(max_ind,arr_shape)
            max_vals[k] = arr[max_inds[k,0],max_inds[k,1]]
            max_vals[k] = arr[max_inds[k,0],max_inds[k,1]]
            arr[max(0,max_inds[k,0]-ix):min(nx,max_inds[k,0]+ix),
                max(0,max_inds[k,1]-iy):min(ny,max_inds[k,1]+iy)] = 0
            k+=1
            
    else:
        max_inds = np.zeros((n,2),numba.int32)
        max_vals = np.zeros(n,numba.float64)
        k = 0
        while np.max(arr) > min_val and k < n:
            max_ind = np.argmax(arr)
            max_inds[k,:] = unravel_index(max_ind,arr_shape)
            max_vals[k] = arr[max_inds[k,0],max_inds[k,1]]
            arr[max(0,max_inds[k,0]-ix):min(nx,max_inds[k,0]+ix),
                max(0,max_inds[k,1]-iy):min(ny,max_inds[k,1]+iy)] = 0
            k+=1
            
    return max_vals[:k], max_inds[:k,:]


@njit
def gen_circ(r,c,n,xlims=None,ylims=None):
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
    
    theta = np.linspace(0,2*pi,n)
    pts = np.zeros((n,2),np.float64)
    cx = c[0]
    cy = c[1]
    for k in prange(n):
        pts[k,0] = r*cos(theta[k]) + cx
        pts[k,1] = r*sin(theta[k]) + cy
        
    if xlims is not None:
        xm = pts[:,0] < xlims[0]
        xM = pts[:,0] > xlims[1]
        maskx = ~np.logical_or(xm,xM)
        pts = pts[maskx,:]
    if ylims is not None:
        ym = pts[:,0] < ylims[0]
        yM = pts[:,0] > ylims[1]
        masky = ~np.logical_or(ym,yM)
        pts = pts[masky,:]
        
    return pts



@njit
def gen_filled_circ(r,n,alpha=3.0,c=np.array([0.0,0.0]),xlims=None,ylims=None):
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
    phi = 0.5*(1 + 5**0.5)
    cd = 1/(phi**2)
    ar = round(alpha*n**0.5)
    x = np.zeros(n,np.float64)
    y = np.zeros(n,np.float64)
    for k in range(1,n+1):
        theta = 2*pi*k*cd
        if k > n - ar:
            radius = r
        else:
            radius = r*((k - 0.5)**0.5)/(n - (ar + 1)/2)**0.5
            
        x[k-1] = radius*cos(theta) + c[0]
        y[k-1] = radius*sin(theta) + c[1]
        
    if xlims is not None:
        xm = x < xlims[0]
        xM = x > xlims[1]
        maskx = ~np.logical_or(xm,xM)
        x = x[maskx]
        y = y[maskx]
    if ylims is not None:
        ym = y < ylims[0]
        yM = y > ylims[1]
        masky = ~np.logical_or(ym,yM)
        x = x[masky]
        y = y[masky]
        
    pts = np.column_stack((x,y))    
    return pts
    
    
@njit
def gen_filled_circ_radius(r,n,alpha=3.0,c=np.array([0.0,0.0]),xlims=None,ylims=None):
    """
    Generate points filling a circle with radius r and center c. Uses the sunflower
    seed arangement. Also returns radius.

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
        array containing radius of earch point from center c.

    """
    phi = 0.5*(1 + 5**0.5)
    cd = 1/(phi**2)
    ar = round(alpha*n**0.5)
    x = np.zeros(n,np.float64)
    y = np.zeros(n,np.float64)
    radius = np.zeros(n,np.float64)
    for k in range(1,n+1):
        theta = 2*pi*k*cd
        if k > n - ar:
            radius[k-1] = r
        else:
            radius[k-1] = r*((k - 0.5)**0.5)/(n - (ar + 1)/2)**0.5
            
        x[k-1] = radius[k-1]*cos(theta) + c[0]
        y[k-1] = radius[k-1]*sin(theta) + c[1]
        
    if xlims is not None:
        xm = x < xlims[0]
        xM = x > xlims[1]
        maskx = ~np.logical_or(xm,xM)
        x = x[maskx]
        y = y[maskx]
        radius = radius[maskx]
    if ylims is not None:
        ym = y < ylims[0]
        yM = y > ylims[1]
        masky = ~np.logical_or(ym,yM)
        x = x[masky]
        y = y[masky]
        radius = radius[masky]
        
    pts = np.column_stack((x,y))    
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
    for k in prange(npts-1):
        p0 = pts[k,:]
        p1 = pts[k+1,:]
        arclength_ += ((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)**0.5
        
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
    arclength_ = np.zeros(npts,numba.float64)
    arclength_[0] = 0.0
    for k in range(1,npts):
        p0 = pts[k-1,:]
        p1 = pts[k,:]
        arclength_[k] = ((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)**0.5 + arclength_[k-1]
        
    return arclength_


def interp_curve(curve,n,s=0,k=3,per=0):
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
    
    tck, u = splprep([curve[:,0], curve[:,1]], k=k, s=s, per=per)
    xi, yi = splev(np.linspace(0, 1, n), tck)
    
    curvei = np.column_stack((xi,yi))
    return curvei


@njit
def wn_pt_in_poly(polygon,point):
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
    dx1 = polygon[0,0] - ptx
    dy1 = polygon[0,1] - pty
    below1 = dy1 <= 0
    for k in range(1,n):
        dx0 = dx1
        dy0 = dy1
        below0 = below1
        dx1 = polygon[k,0] - ptx
        dy1 = polygon[k,1] - pty
        below1 = dy1 <= 0
        is_left = dx0*dy1 - dx1*dy0 > 0
        wn += (below0 & (below1 ^ 1) & is_left) - (below1 & (below0 ^ 1) & ~is_left)

    return wn

@njit
def pts_in_poly(polygon,pts):
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
        if point from pts is found inside polygon, its index is returned, if not, -1 is returned.

    """

    for k in range(len(pts)):
        pt = pts[k,:]  
        
        if wn_pt_in_poly(polygon,pt):
            return k
        
    return -1


@njit
def pts_in_poly_mask(polygon,pts):
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
    mask = np.zeros((npts,),np.bool_)
    for k in range(npts):
        mask[k] = wn_pt_in_poly(polygon,pts[k,:])
        
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
    shape = np.zeros(nvecs,np.int32)
    for k in range(nvecs):
        shape[k] = len(vecs[k])
    npts = np.prod(shape)
    prod = np.zeros((npts,nvecs),dtype)
    
    if nvecs > 2:
        for k in prange(nvecs-1):
            cl = np.prod(shape[-1:k:-1])
            arr = vecs[k]
            shapek = shape[k]
            for j in range(shapek):
                prod[j*cl:(j+1)*cl,k] = arr[j]
                
            full_len = shapek*cl
            for i in range(np.prod(shape[:k])-1):
                prod[(i+1)*full_len:(i+2)*full_len,k] = prod[:full_len,k]
    else:
        cl = shape[1]
        arr = vecs[0]
        for j in prange(shape[0]):
            prod[j*cl:(j+1)*cl,0] = arr[j]
                        
    nlast = shape[-1]
    arr = vecs[-1]
    for j in prange(np.prod(shape[:-1])):
        prod[j*nlast:(j+1)*nlast,-1] = arr 

    return prod

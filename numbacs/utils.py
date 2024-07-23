import numpy as np
from numba import njit, prange
import numba

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
        DESCRIPTION.
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
def curl(f,dx,dy):
    """
    Compute curl of vector field defined by f with underlying grid spacing dx and dy.

    Parameters
    ----------
    f : np.ndarray, shape = (nx,ny,2)
        array containing x and y components of vector field.
    dx : float
        grid spacing in x-direction.
    dy : int
        grid spacing in y-direction.

    Returns
    -------
    curlf : np.ndarray, shape = (nx,ny)
        array containing values of curl of f.

    """
    
    nx,ny = f.shape[:-1]
    curlf = np.zeros((nx,ny),numba.float64)
    for i in prange(nx):
        if i == 0:
            dxdir = 'f'
        elif i == nx-1:
            dxdir = 'b'
        else:
            dxdir = 'c'
        for j in range(ny):
            if j == 0:
               dydir = 'f'
            elif j == ny-1:
                dydir = 'b'
            else:
                dydir = 'c'
            
            dfydx = finite_diff_2D(f[:,:,1],i,j,dx,0,dxdir)
            dfxdy = finite_diff_2D(f[:,:,0],i,j,dy,1,dydir)
            
            curlf[i,j] = dfydx - dfxdy
    
    return curlf
    
    
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


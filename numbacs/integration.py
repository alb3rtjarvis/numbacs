import numpy as np
from numbalsoda import dop853, lsoda
from numba import njit, prange
import numba

parallel_flag=True

@njit(parallel=parallel_flag)
def flowmap(funcptr,t0,T,pts,params,method='dop853',rtol=1e-6,atol=1e-8):
    """
    Computes the flow map of the ode defined by funcptr where funcptr is a pointer to a C callback
    created within numbalsoda using the ``@cfunc`` decorator. Flow map is computed from initial
    conditions given by pts where pts has dim (npts,2). t0 denotes initial time and T denotes
    integration time.

    Parameters
    ----------
    funcptr : int
        pointer to C callback.
    t0 : float
        intial time.
    T : float
        integration time.
    pts : np.ndarray, shape = (npts,2)
        array of points to be integrated.
    params : np.ndarray, shape = (nprms,)
        array of parameters to be passed to the ode function defined by funcptr.
    method : str, optional
        method to be used by numbalsoda to solve ode. The default is 'dop853'.
    rtol : float, optional
        relative tolerance for ode solver. The default is 1e-6.
    atol : float, optional
        absolute tolerance for ode solver. The default is 1e-8.

    Returns
    -------
    flowmap : np.ndarray, shape = (npts,2)
        array containing final position of particles pts after integration from [t0,t0+T].

    """
    flowmap = np.zeros(pts.shape,numba.float64)
    t_eval = params[0]*np.linspace(t0,t0+T,2)
    if method == 'dop853':
        for i in prange(pts.shape[0]):
            flowmap_tmp,success = dop853(funcptr, np.array([pts[i,0],pts[i,1]]),
                                         t_eval,rtol=rtol,atol=atol,data=params)
            flowmap[i,:] = flowmap_tmp[-1,:]
    elif method == 'lsoda':
        for i in prange(pts.shape[0]):
            flowmap_tmp,success = lsoda(funcptr, np.array([pts[i,0],pts[i,1]]),
                                         t_eval,rtol=rtol,atol=atol,data=params)
            flowmap[i,:] = flowmap_tmp[-1,:]
    
    return flowmap


@njit(parallel=parallel_flag)
def flowmap_n(funcptr,t0,T,pts,params,method='dop853',n=2,rtol=1e-6,atol=1e-8):
    """
    Computes the flow map of the ode defined by funcptr where funcptr is a pointer to a C callback
    created within numbalsoda using the ``@cfunc`` decorator. Flow map is computed from initial
    conditions given by pts where pts has dim (npts,2). t0 denotes initial time and T denotes
    integration time.

    Parameters
    ----------
    funcptr : int
        pointer to C callback.
    t0 : float
        intial time.
    T : float
        integration time.
    pts : np.ndarray, shape = (npts,2)
        array of points to be integrated.
    params : np.ndarray, shape = (nprms,)
        array of parameters to be passed to the ode function defined by funcptr.
    method : str, optional
        method to be used by numbalsoda to solve ode. The default is 'dop853'.
    n : int, optional
        number of points to return the flowmap at (including initial condition). The default is 2.
    rtol : float, optional
        relative tolerance for ode solver. The default is 1e-6.
    atol : float, optional
        absolute tolerance for ode solver. The default is 1e-8.

    Returns
    -------
    flowmap : np.ndarray, shape = (npts,n,2)
        array containing n positions of particles pts after integration from [t0,t0+T].
    t_eval : np.ndarray, shape = (n,)
        array containing times the flowmap is returned at.

    """
    npts = len(pts)
    flowmap = np.zeros((npts,n,2),numba.float64)
    t_eval = params[0]*np.linspace(t0,t0+T,n)
    if method == 'dop853':
        for i in prange(pts.shape[0]):
            flowmap_tmp,success = dop853(funcptr, np.array([pts[i,0],pts[i,1]]),
                                         t_eval,rtol=rtol,atol=atol,data=params)
            flowmap[i,:,:] = flowmap_tmp
    elif method == 'lsoda':
        for i in prange(pts.shape[0]):
            flowmap_tmp,success = lsoda(funcptr, np.array([pts[i,0],pts[i,1]]),
                                         t_eval,rtol=rtol,atol=atol,data=params)
            flowmap[i,:,:] = flowmap_tmp
    
    return flowmap, params[0]*t_eval


@njit(parallel=parallel_flag)
def flowmap_grid_2D(funcptr,t0,T,x,y,params,method='dop853',rtol=1e-6,atol=1e-8):
    """
    Computes the flow map at the final time of the ode defined by funcptr where funcptr is a
    pointer to a C callback created within numba using the ``@cfunc`` decorator. Flow map is
    computed over the grid defined by x,y. t0 denotes initial time and T denotes
    integration time.

    Parameters
    ----------
    funcptr : int
        pointer to C callback.
    t0 : float
        intial time.
    T : float
        integration time.
    x : np.ndarray, shape = (nx,)
        array containing x-values.
    y : np.ndarray, shape = (ny,)
        array containing y-values.
    params : np.ndarray, shape = (nprms,)
        array of parameters to be passed to the ode function defined by funcptr.
    method : str, optional
        method to be used by numbalsoda to solve ode. The default is 'dop853'.
    rtol : float, optional
        relative tolerance for ode solver. The default is 1e-6.
    atol : float, optional
        absolute tolerance for ode solver. The default is 1e-8.

    Returns
    -------
    flowmap : np.ndarray, shape = (nx,ny,2)
        array containing final positions of particles pts after integration from [t0,t0+T].

    """

    nx,ny = len(x),len(y)
    flowmap = np.zeros((nx,ny,2),numba.float64)
    t_eval = params[0]*np.linspace(t0,t0+T,2)
    if method == 'dop853':
        for i in prange(nx):
            for j in range(ny):
                flowmap_tmp,success = dop853(funcptr, np.array([x[i],y[j]]),
                                             t_eval,rtol=rtol,atol=atol,data=params)
                flowmap[i,j,:] = flowmap_tmp[-1,:]
    elif method == 'lsoda':
        for i in prange(nx):
            for j in range(ny):
                flowmap_tmp,success = lsoda(funcptr, np.array([x[i],y[j]]),
                                            t_eval,rtol=rtol,atol=atol,data=params)
                flowmap[i,j,:] = flowmap_tmp[-1,:]
    
    return flowmap



@njit(parallel=parallel_flag)
def flowmap_grid_ND(funcptr,t0,T,IC_flat,ndims,params,method='dop853',rtol=1e-6,atol=1e-8):
    """
    Computes the flow map at the final time of the ode defined by funcptr where funcptr is a
    pointer to a C callback created within numba using the ``@cfunc`` decorator. Flow map is
    computed over the grid defined by IC_flat where IC_flat has shape
    (nx_1*nx_2*...*nx_ndims*ndims). t0 denotes initial time and T denotes integration time.

    Parameters
    ----------
    funcptr : int
        pointer to C callback.
    t0 : float
        intial time.
    T : float
        integration time.
    IC_flat : np.ndarray, shape = (nx_1*nx_2*...*nx_ndims*ndims,)
        flattened array of initial conditions, can be obtained by IC.flatten() or IC.ravel().
    ndims : int
        number of dimensions.
    params : np.ndarray, shape = (nprms,)
        array of parameters to be passed to the ode function defined by funcptr.
    method : str, optional
        method to be used by numbalsoda to solve ode. The default is 'dop853'.
    rtol : float, optional
        relative tolerance for ode solver. The default is 1e-6.
    atol : float, optional
        absolute tolerance for ode solver. The default is 1e-8.

    Returns
    -------
    flowmap : np.ndarray, shape = (nx_1*nx_2*...*nx_ndims,ndims)
        array containing final positions of particles IC_flat after integration from [t0,t0+T].

    """
    npts = int(len(IC_flat)/ndims)
    flowmap = np.zeros((npts,ndims),numba.float64)
    t_eval = params[0]*np.linspace(t0,t0+T,2)
    if method == 'dop853':
        for k in prange(npts):
            flowmap_tmp,success = dop853(funcptr, IC_flat[k*ndims:(k+1)*ndims],
                                         t_eval,rtol=rtol,atol=atol,data=params)
            flowmap[k,:] = flowmap_tmp[-1,:]
    elif method == 'lsoda':
        for k in prange(npts):
            flowmap_tmp,success = lsoda(funcptr, IC_flat[k*ndims:(k+1)*ndims],
                                        t_eval,rtol=rtol,atol=atol,data=params)
            flowmap[k,:] = flowmap_tmp[-1,:]
    
    return flowmap



@njit(parallel=parallel_flag)
def flowmap_aux_grid_2D(funcptr,t0,T,x,y,params,h=1e-5,eig_main=True,compute_edge=True,
                        method='dop853',rtol=1e-6,atol=1e-8):
    """
    Computes the flow map at the final time of the ode defined by funcptr where funcptr is
    a pointer to a C callback created within numba using the ``@cfunc`` decorator. Flow map
    is computed over the aux grid defined by np.meshgrid(x,y) +-h. t0 denotes initial time
    and T denotes integration time.

    Parameters
    ----------
    funcptr : int
        pointer to C callback.
    t0 : float
        intial time.
    T : float
        integration time.
    x : np.ndarray, shape = (nx,)
        array containing x-values.
    y : np.ndarray, shape = (ny,)
        array containing y-values.
    params : np.ndarray, shape = (nprms,)
        array of parameters to be passed to the ode function defined by funcptr.
    h : float, optional
        spacing for the auxilary grid. The default is 1e-5.
    eig_main : boolean, optional
        flag which determines if integration is performed on main grid in addition
        to aux grid, used if eigenvalues are to be computed from main grid in 'C_eig_2D'.
        The default is True.
    compute_edge : boolean, optional
        flag which determines if integration is performed on edge of grid, set to True
        if ode is defined outside of x,y, False if it is not . The default is True.
    method : str, optional
        method to be used by numbalsoda to solve ode. The default is 'dop853'.
    rtol : float, optional
        relative tolerance for ode solver. The default is 1e-6.
    atol : float, optional
        absolute tolerance for ode solver. The default is 1e-8.

    Returns
    -------
    flowmap_aux : np.ndarray, shape = (nx,ny,n_aux,2)
        array containing final positions of particles (including aux grid) after
        integration from [t0,t0+T].

    """

    nx,ny = len(x),len(y)
    t_eval = params[0]*np.linspace(t0,t0+T,2)
    if eig_main:
        n_aux = 5
        aux_grid = np.array([[h,0.],[-h,0.],[0.,h],[0.,-h],[0.,0.]])
        flowmap_aux = np.zeros((nx,ny,n_aux,2),numba.float64)
        if method == 'dop853':
            if compute_edge:
                for i in prange(nx):
                    if i == 0 or i == nx-1:
                        for j in range(ny):
                            flowmap_tmp,success = dop853(funcptr, 
                                                         np.array([x[i], y[j]])+
                                                         aux_grid[-1,:],t_eval,
                                                          rtol=rtol,atol=atol,data=params)
                            flowmap_aux[i,j,-1,:] = flowmap_tmp[-1,:]
                    else:
                        for j in range(ny):
                             if j == 0 or j == ny-1:
                                 flowmap_tmp,success = dop853(funcptr, 
                                                              np.array([x[i], y[j]])+
                                                              aux_grid[-1,:],t_eval,
                                                              rtol=rtol,atol=atol,data=params)
                                 flowmap_aux[i,j,-1,:] = flowmap_tmp[-1,:]
                             else:
                                 for k in range(n_aux):
                                     flowmap_tmp,success = dop853(funcptr, 
                                                                  np.array([x[i], y[j]])+
                                                                  aux_grid[k,:],t_eval,
                                                                  rtol=rtol,atol=atol,data=params)
                                     flowmap_aux[i,j,k,:] = flowmap_tmp[-1,:]
            else:
                 for i in prange(1,nx-1):
                     for j in prange(1,ny-1):
                        for k in range(n_aux):
                            flowmap_tmp,success = dop853(funcptr, 
                                                         np.array([x[i], y[j]])+
                                                         aux_grid[k,:],t_eval,
                                                         rtol=rtol,atol=atol,data=params)
                            flowmap_aux[i,j,k,:] = flowmap_tmp[-1,:]
        elif method == 'lsoda':
            if compute_edge:
                for i in prange(nx):
                    if i == 0 or i == nx-1:
                        for j in range(ny):
                            flowmap_tmp,success = lsoda(funcptr, 
                                                        np.array([x[i], y[j]])+
                                                        aux_grid[-1,:],t_eval,
                                                        rtol=rtol,atol=atol,data=params)
                            flowmap_aux[i,j,-1,:] = flowmap_tmp[-1,:]
                    else:
                        for j in range(ny):
                             if j == 0 or j == ny-1:
                                 flowmap_tmp,success = lsoda(funcptr,
                                                             np.array([x[i], y[j]])+
                                                             aux_grid[-1,:],t_eval,
                                                             rtol=rtol,atol=atol,data=params)
                                 flowmap_aux[i,j,-1,:] = flowmap_tmp[-1,:]
                             else:
                                 for k in range(n_aux):
                                     flowmap_tmp,success = lsoda(funcptr,
                                                                 np.array([x[i], y[j]])+
                                                                 aux_grid[k,:],t_eval,
                                                                 rtol=rtol,atol=atol,data=params)
                                     flowmap_aux[i,j,k,:] = flowmap_tmp[-1,:]
            else:
                 for i in prange(1,nx-1):
                     for j in prange(1,ny-1):
                        for k in range(n_aux):
                            flowmap_tmp,success = lsoda(funcptr,
                                                        np.array([x[i], y[j]])+
                                                        aux_grid[k,:], t_eval,
                                                        rtol=rtol,atol=atol,data=params)
                            flowmap_aux[i,j,k,:] = flowmap_tmp[-1,:]
                     
       
    else:
        n_aux = 4
        aux_grid = np.array([[h,0.],[-h,0.],[0.,h],[0.,-h]])
        flowmap_aux = np.zeros((nx,ny,n_aux,2),numba.float64)
        if compute_edge == True:
            xrange = np.array([0,nx],numba.int32)
            yrange = np.array([0,ny],numba.int32)
        else:
            xrange = np.array([1,nx-1],numba.int32)
            yrange = np.array([1,ny-1],numba.int32)
        if method == 'dop853':
            for i in prange(xrange[0],xrange[1]):
                for j in range(yrange[0],yrange[1]):
                    for k in range(n_aux):
                        flowmap_tmp,success = dop853(funcptr, 
                                                     np.array([x[i], y[j]])+aux_grid[k,:],
                                                     t_eval,
                                                     rtol=rtol,atol=atol,data=params)
                        flowmap_aux[i,j,k,:] = flowmap_tmp[-1,:]
        elif method == 'lsoda':
            for i in prange(xrange[0],xrange[1]):
                for j in range(yrange[0],yrange[1]):
                    for k in range(n_aux):
                        flowmap_tmp,success = lsoda(funcptr, 
                                                    np.array([x[i], y[j]])+aux_grid[k,:],
                                                    t_eval,
                                                    rtol=rtol,atol=atol,data=params)
                        flowmap_aux[i,j,k,:] = flowmap_tmp[-1,:]
    
    return flowmap_aux


@njit(parallel=parallel_flag)
def flowmap_n_grid_2D(funcptr,t0,T,x,y,params,n=50,method='dop853',rtol=1e-6,atol=1e-8):
    """
    

    Parameters
    ----------
    funcptr : int
        pointer to C callback.
    t0 : float
        intial time.
    T : float
        integration time.
    x : np.ndarray, shape = (nx,)
        array containing x-values.
    y : np.ndarray, shape = (ny,)
        array containing y-values.
    params : np.ndarray, shape = (nprms,)
        array of parameters to be passed to the ode function defined by funcptr.
        DESCRIPTION.
    n : int, optional
        number of points to return the flowmap at. The default is 50.
    method : str, optional
        method to be used by numbalsoda to solve ode. The default is 'dop853'.
    rtol : float, optional
        relative tolerance for ode solver. The default is 1e-6.
    atol : float, optional
        absolute tolerance for ode solver. The default is 1e-8.

    Returns
    -------
    flowmap : np.ndarray, shape = (nx,ny,n,2)
        array containing n positions of particles after integration from [t0,t0+T].
    np.ndarray, shape = (n,)
        array containing t-values at which flowmap is being returned.

    """

    nx,ny = len(x), len(y)
    t_eval = params[0]*np.linspace(t0,t0+T,n)
    flowmap = np.zeros((nx,ny,n,2),numba.float64)
    if method == 'dop853':
        for i in prange(nx):
            for j in range(ny):
                flowmap_tmp,success = dop853(funcptr, np.array([x[i], y[j]]),
                                             t_eval,rtol=rtol,atol=atol,data=params)
                flowmap[i,j,:,:] = flowmap_tmp
    elif method == 'lsoda':
        for i in prange(nx):
            for j in range(ny):
                flowmap_tmp,success = lsoda(funcptr, np.array([x[i], y[j]]),
                                             t_eval,rtol=rtol,atol=atol,data=params)
                flowmap[i,j,:,:] = flowmap_tmp
    
    return flowmap, params[0]*t_eval



@njit(parallel=parallel_flag)
def flowmap_n_grid_ND(funcptr,t0,T,IC_flat,ndims,params,n=50,method='dop853',rtol=1e-6,atol=1e-8):
    """
    Computes the flow map at the n times of the ode defined by funcptr where funcptr is a
    pointer to a C callback created within numba using the ``@cfunc`` decorator. Flow map is
    computed over the grid defined by IC_flat where IC_flat has shape
    (nx_1*nx_2*...*nx_ndims*ndims). t0 denotes initial time and T denotes integration time.
    

    Parameters
    ----------
    funcptr : int
        pointer to C callback.
    t0 : float
        intial time.
    T : float
        integration time.
    IC_flat : np.ndarray, shape = (nx_1*nx_2*...*nx_ndims*ndims,)
        flattened array of initial conditions, can be obtained by IC.flatten() or IC.ravel().
    ndims : int
        number of dimensions.
    params : np.ndarray, shape = (nprms,)
        array of parameters to be passed to the ode function defined by funcptr.
    n : int, optional
        number of points to return the flowmap at. The default is 50.
    method : str, optional
        method to be used by numbalsoda to solve ode. The default is 'dop853'.
    rtol : float, optional
        relative tolerance for ode solver. The default is 1e-6.
    atol : float, optional
        absolute tolerance for ode solver. The default is 1e-8.
        
    Returns
    -------
    flowmap : np.ndarray, shape = (nx_1*nx_2*...*nx_ndims,n,ndims)
        array containing n positions of particles after integration from [t0,t0+T].
    np.ndarray, shape = (n,)
        array containing t-values at which flowmap is being returned.

    """

    t_eval = params[0]*np.linspace(t0,t0+T,n)
    npts = int(len(IC_flat)/ndims)
    flowmap = np.zeros((npts,n,ndims),numba.float64)
    if method == 'dop853':
        for k in prange(npts):
            flowmap_tmp,success = dop853(funcptr, IC_flat[k*ndims:(k+1)*ndims],
                                          t_eval ,rtol=rtol,atol=atol,data=params)
            flowmap[k,:,:] = flowmap_tmp
    elif method == 'lsoda':
        for k in prange(npts):
            flowmap_tmp,success = lsoda(funcptr, IC_flat[k*ndims:(k+1)*ndims],
                                          t_eval ,rtol=rtol,atol=atol,data=params)
            flowmap[k,:,:] = flowmap_tmp
    
    return flowmap, params[0]*t_eval

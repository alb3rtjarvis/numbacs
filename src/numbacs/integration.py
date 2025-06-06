import numpy as np
from numbalsoda import dop853, lsoda
from numba import njit, prange, float64, int32
from interpolation.splines import eval_linear, extrap_options as xto


@njit(parallel=True)
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

    flowmap = np.zeros(pts.shape,float64)
    t_eval = params[0]*np.linspace(t0,t0+T,2)
    if method.lower() == 'dop853':
        for i in prange(pts.shape[0]):
            flowmap_tmp,success = dop853(funcptr, np.array([pts[i,0],pts[i,1]]),
                                         t_eval,rtol=rtol,atol=atol,data=params)
            flowmap[i,:] = flowmap_tmp[-1,:]
    elif method.lower() == 'lsoda':
        for i in prange(pts.shape[0]):
            flowmap_tmp,success = lsoda(funcptr, np.array([pts[i,0],pts[i,1]]),
                                         t_eval,rtol=rtol,atol=atol,data=params)
            flowmap[i,:] = flowmap_tmp[-1,:]

    return flowmap


@njit(parallel=True)
def flowmap_n(funcptr,t0,T,pts,params,method='dop853',n=2,rtol=1e-6,atol=1e-8):
    """
    Computes the flow map of the ode defined by funcptr where funcptr is a pointer to a C callback
    created within numbalsoda using the ``@cfunc`` decorator. Flow map is computed from initial
    conditions given by pts where pts has dim (npts,2). t0 denotes initial time and T denotes
    integration time, flowmap is returned at n times in [t0,t0+T] (inclusive).

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
    flowmap = np.zeros((npts,n,2),float64)
    t_eval = params[0]*np.linspace(t0,t0+T,n)
    if method.lower() == 'dop853':
        for i in prange(pts.shape[0]):
            flowmap_tmp,success = dop853(funcptr, np.array([pts[i,0],pts[i,1]]),
                                         t_eval,rtol=rtol,atol=atol,data=params)
            flowmap[i,:,:] = flowmap_tmp
    elif method.lower() == 'lsoda':
        for i in prange(pts.shape[0]):
            flowmap_tmp,success = lsoda(funcptr, np.array([pts[i,0],pts[i,1]]),
                                         t_eval,rtol=rtol,atol=atol,data=params)
            flowmap[i,:,:] = flowmap_tmp

    return flowmap, params[0]*t_eval


@njit(parallel=True)
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
    flowmap = np.zeros((nx,ny,2),float64)
    t_eval = params[0]*np.linspace(t0,t0+T,2)
    if method.lower() == 'dop853':
        for i in prange(nx):
            for j in range(ny):
                flowmap_tmp,success = dop853(funcptr, np.array([x[i],y[j]]),
                                             t_eval,rtol=rtol,atol=atol,data=params)
                flowmap[i,j,:] = flowmap_tmp[-1,:]
    elif method.lower() == 'lsoda':
        for i in prange(nx):
            for j in range(ny):
                flowmap_tmp,success = lsoda(funcptr, np.array([x[i],y[j]]),
                                            t_eval,rtol=rtol,atol=atol,data=params)
                flowmap[i,j,:] = flowmap_tmp[-1,:]

    return flowmap



@njit(parallel=True)
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
    flowmap = np.zeros((npts,ndims),float64)
    t_eval = params[0]*np.linspace(t0,t0+T,2)
    if method.lower() == 'dop853':
        for k in prange(npts):
            flowmap_tmp,success = dop853(funcptr, IC_flat[k*ndims:(k+1)*ndims],
                                         t_eval,rtol=rtol,atol=atol,data=params)
            flowmap[k,:] = flowmap_tmp[-1,:]
    elif method.lower() == 'lsoda':
        for k in prange(npts):
            flowmap_tmp,success = lsoda(funcptr, IC_flat[k*ndims:(k+1)*ndims],
                                        t_eval,rtol=rtol,atol=atol,data=params)
            flowmap[k,:] = flowmap_tmp[-1,:]

    return flowmap



@njit(parallel=True)
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
        flowmap_aux = np.zeros((nx,ny,n_aux,2),float64)
        if method.lower() == 'dop853':
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
        elif method.lower() == 'lsoda':
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
        flowmap_aux = np.zeros((nx,ny,n_aux,2),float64)
        if compute_edge:
            xrange = np.array([0,nx],int32)
            yrange = np.array([0,ny],int32)
        else:
            xrange = np.array([1,nx-1],int32)
            yrange = np.array([1,ny-1],int32)
        if method.lower() == 'dop853':
            for i in prange(xrange[0],xrange[1]):
                for j in range(yrange[0],yrange[1]):
                    for k in range(n_aux):
                        flowmap_tmp,success = dop853(funcptr,
                                                     np.array([x[i], y[j]])+aux_grid[k,:],
                                                     t_eval,
                                                     rtol=rtol,atol=atol,data=params)
                        flowmap_aux[i,j,k,:] = flowmap_tmp[-1,:]
        elif method.lower() == 'lsoda':
            for i in prange(xrange[0],xrange[1]):
                for j in range(yrange[0],yrange[1]):
                    for k in range(n_aux):
                        flowmap_tmp,success = lsoda(funcptr,
                                                    np.array([x[i], y[j]])+aux_grid[k,:],
                                                    t_eval,
                                                    rtol=rtol,atol=atol,data=params)
                        flowmap_aux[i,j,k,:] = flowmap_tmp[-1,:]

    return flowmap_aux


@njit(parallel=True)
def flowmap_n_grid_2D(funcptr,t0,T,x,y,params,n=50,method='dop853',rtol=1e-6,atol=1e-8):
    """
    Computes the flow map of the ode defined by funcptr where funcptr is a
    pointer to a C callback created within numba using the ``@cfunc`` decorator. Flow map is
    computed over the grid defined by x,y. t0 denotes initial time and T denotes
    integration time, flowmap is returned at n times in [t0,t0+T] (inclusive).

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
    flowmap = np.zeros((nx,ny,n,2),float64)
    if method.lower() == 'dop853':
        for i in prange(nx):
            for j in range(ny):
                flowmap_tmp,success = dop853(funcptr, np.array([x[i], y[j]]),
                                             t_eval,rtol=rtol,atol=atol,data=params)
                flowmap[i,j,:,:] = flowmap_tmp
    elif method.lower() == 'lsoda':
        for i in prange(nx):
            for j in range(ny):
                flowmap_tmp,success = lsoda(funcptr, np.array([x[i], y[j]]),
                                             t_eval,rtol=rtol,atol=atol,data=params)
                flowmap[i,j,:,:] = flowmap_tmp

    return flowmap, params[0]*t_eval



@njit(parallel=True)
def flowmap_n_grid_ND(funcptr,t0,T,IC_flat,ndims,params,n=50,method='dop853',rtol=1e-6,atol=1e-8):
    """
    Computes the flow map at the n times of the ode defined by funcptr where funcptr is a
    pointer to a C callback created within numba using the ``@cfunc`` decorator. Flow map is
    computed over the grid defined by IC_flat where IC_flat has shape
    (nx_1*nx_2*...*nx_ndims*ndims). t0 denotes initial time and T denotes integration time,
    flowmap is returned at n times in [t0,t0+T] (inclusive).


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
    flowmap = np.zeros((npts,n,ndims),float64)
    if method.lower() == 'dop853':
        for k in prange(npts):
            flowmap_tmp,success = dop853(funcptr, IC_flat[k*ndims:(k+1)*ndims],
                                          t_eval ,rtol=rtol,atol=atol,data=params)
            flowmap[k,:,:] = flowmap_tmp
    elif method.lower() == 'lsoda':
        for k in prange(npts):
            flowmap_tmp,success = lsoda(funcptr, IC_flat[k*ndims:(k+1)*ndims],
                                          t_eval ,rtol=rtol,atol=atol,data=params)
            flowmap[k,:,:] = flowmap_tmp

    return flowmap, params[0]*t_eval


def flowmap_composition(flowmaps,grid,nT):
    """
    Interpolation for flowmap composition method. Returns composed flowmap from
    intermediate flowmaps.

    Parameters
    ----------
    flowmaps : np.ndarray, shape = (nT,nx,ny,2)
        flowmaps for each (k+1)*t0 to (k+1)*t0+h for k in {0,...,nT-1}.
    grid : tuple
        tuple containing endpoints and number of points for each dimension.
    nT : int
        number of intermediate flowmaps.

    Returns
    -------
    composed_flowmap : np.ndarray, shape = (nx,ny,2)
        composed flowmap from t0 to t0 + T.

    """
    nx,ny = grid[0][2],grid[1][2]
    composed_flowmap = np.zeros((nx,ny,2),np.float64)
    pts = np.column_stack((flowmaps[0,:,:,0].ravel(),flowmaps[0,:,:,1].ravel()))
    for k in range(1,nT-1):
        fx = eval_linear(grid,flowmaps[k,:,:,0],pts,xto.CONSTANT)
        fy = eval_linear(grid,flowmaps[k,:,:,1],pts,xto.CONSTANT)
        pts = np.column_stack((fx,fy))

    composed_flowmap[:,:,0] = eval_linear(grid,flowmaps[-1,:,:,0],pts,xto.CONSTANT).reshape(nx,ny)
    composed_flowmap[:,:,1] = eval_linear(grid,flowmaps[-1,:,:,1],pts,xto.CONSTANT).reshape(nx,ny)

    return composed_flowmap



def flowmap_composition_initial(funcptr,t0,T,h,x,y,grid,params,**kwargs):
    """
    Initial step for flowmap composition method. Returns first full flowmap
    and collection of intermediate flowmaps to be used in successive steps.

    Parameters
    ----------
    funcptr : int
        pointer to C callback.
    t0 : float
        intial time.
    T : float
        overall integration time.
    h : float
        integration time for each flowmap composition. T/h should be an integer or
        the results will be incorrect.
    x : np.ndarray, shape = (nx,)
        array containing x-values.
    y : np.ndarray, shape = (ny,)
        array containing y-values.
    grid : tuple
        tuple containing endpoints and number of points for each dimension.
    params : np.ndarray, shape = (nprms,)
        array of parameters to be passed to the ode function defined by funcptr.
    **kwargs : str,float
        keyword arguments for flowmap_grid_2D.

    Returns
    -------
    flowmap0 : np.ndarray, shape = (nx,ny,2)
        flowmap from t0 to t0 + T.
    flowmaps : np.ndarray, shape = (nT,nx,ny,2)
        flowmaps for each (k+1)*t0 to (k+1)*t0+h for k in {0,...,nT-1}.
    nT : int
        number of points making up integration span

    """

    nT = abs(round(T/h))
    flowmaps = np.zeros((nT,grid[0][2],grid[1][2],2),np.float64)
    for k in range(nT):
        flowmaps[k,:,:,:] = flowmap_grid_2D(funcptr,t0,h,x,y,params,**kwargs)
        t0 += h
    flowmap0 = flowmap_composition(flowmaps,grid, nT)

    return flowmap0, flowmaps, nT


def flowmap_composition_step(flowmaps,funcptr,t0,h,nT,x,y,grid,params,**kwargs):
    """
    Step for flowmap composition method. Returns full flowmap at current step
    and collection of intermediate flowmaps to be used in successive steps.

    Parameters
    ----------
    flowmaps : np.ndarray, shape = (nT,nx,ny,2)
        flowmaps for each (k+1)*t0 to (k+1)*t0+h for k in {0,...,nT-1}.
    funcptr : int
        pointer to C callback.
    t0 : float
        intial time.
    h : float
        integration time for each flowmap composition. T/h should be an integer or
        the results will be incorrect.
    nT : int
        number of intermediate flowmaps.
    x : np.ndarray, shape = (nx,)
        array containing x-values.
    y : np.ndarray, shape = (ny,)
        array containing y-values.
    grid : tuple
        tuple containing endpoints and number of points for each dimension.
    params : np.ndarray, shape = (nprms,)
        array of parameters to be passed to the ode function defined by funcptr.
    **kwargs : str,float
        keyword arguments for flowmap_grid_2D.

    Returns
    -------
    flowmap_k : np.ndarray, shape = (nx,ny,2)
        composed flowmap from t0 to t0 + T.
    flowmaps : np.ndarray, shape = (nT,nx,ny,2)
        flowmaps for each (k+1)*t0 to (k+1)*t0+h for k in {1,...,nT}.

    """

    flowmaps[:-1,:,:,:] = flowmaps[1:,:,:,:]
    flowmaps[-1,:,:,:] = flowmap_grid_2D(funcptr,t0,h,x,y,params,**kwargs)
    flowmap_k = flowmap_composition(flowmaps, grid, nT)

    return flowmap_k, flowmaps

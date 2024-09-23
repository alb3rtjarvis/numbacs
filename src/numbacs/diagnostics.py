import numpy as np
from math import log
from numba import njit, prange, float64
from .utils import composite_simpsons, unravel_index, finite_diff_ND


@njit(parallel=True)
def ftle_grid_2D(flowmap,T,dx,dy):
    """
    Compute 2D FTLE field from flowmap which is solution of ode over an initial grid defined by x
    and y for integration time T.

    Parameters
    ----------
    flowmap : np.ndarray, shape = (nx,ny,2)
        array containing final positions of trajectories of initial grid from t0 to t0+T.
    T : float
        integration time.
    dx : float
        grid spacing in x-direction.
    dy : float
        grid spacing in y-direction.

    Returns
    -------
    ftle : np.ndarray, shape = (nx,ny)
        array containing ftle values.

    """
    nx,ny = flowmap.shape[:-1]
    ftle = np.zeros((nx,ny),float64)
    absT = abs(T)
    for i in prange(1,nx-1):
        for j in range(1,ny-1):
            dxdx = (flowmap[i+1,j,0] - flowmap[i-1,j,0])/(2*dx)
            dxdy = (flowmap[i,j+1,0] - flowmap[i,j-1,0])/(2*dy)
            dydx = (flowmap[i+1,j,1] - flowmap[i-1,j,1])/(2*dx)
            dydy = (flowmap[i,j+1,1] - flowmap[i,j-1,1])/(2*dy)
            
            off_diagonal = dxdx*dxdy + dydx*dydy
            C = np.array([[dxdx**2 + dydx**2, off_diagonal],
                           [off_diagonal, dxdy**2 + dydy**2]])
            
            max_eig = np.linalg.eigvalsh(C)[-1]
            if max_eig > 1:
                ftle[i,j] = 1/(2*absT)*log(max_eig)
            else: 
                ftle[i,j] = 0
            
    return ftle



@njit(parallel=True)
def C_tensor_2D(flowmap_aux,dx,dy,h=1e-5):
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

    Returns
    -------
    C : np.ndarray, shape = (nx,ny,3)
        array containing C11, C12, C22 components of Cauchy Green tensor.

    """
    nx,ny = flowmap_aux.shape[:2]
    C = np.zeros((nx,ny,3),float64)
    for i in prange(2,nx-2):
        for j in range(2,ny-2):
            dxdx_aux = (flowmap_aux[i,j,0,0] - flowmap_aux[i,j,1,0])/(2*h)
            dxdy_aux = (flowmap_aux[i,j,2,0] - flowmap_aux[i,j,3,0])/(2*h)
            dydx_aux = (flowmap_aux[i,j,0,1] - flowmap_aux[i,j,1,1])/(2*h)
            dydy_aux = (flowmap_aux[i,j,2,1] - flowmap_aux[i,j,3,1])/(2*h)
            
            C[i,j,:] = np.array([dxdx_aux**2 + dydx_aux**2, 
                               dxdx_aux*dxdy_aux + dydx_aux*dydy_aux, 
                               dxdy_aux**2 + dydy_aux**2])
            
                
    return C


@njit(parallel=True)
def C_eig_aux_2D(flowmap_aux,dx,dy,h=1e-5,eig_main=True):
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

    Returns
    -------
    eigvals : np.ndarray, shape = (nx,ny,2)
        array containing eigenvalues values.
    eigvecs : np.ndarray, shape = (nx,ny,2,2)
        array containing eigenvectors.

    """

    nx,ny = flowmap_aux.shape[:-2]
    eigvals = np.zeros((nx,ny,2),float64)
    eigvecs = np.zeros((nx,ny,2,2),float64)
    if eig_main:
        for i in prange(2,nx-2):
            for j in range(2,ny-2):
                dxdx_aux = (flowmap_aux[i,j,0,0] - flowmap_aux[i,j,1,0])/(2*h)
                dxdy_aux = (flowmap_aux[i,j,2,0] - flowmap_aux[i,j,3,0])/(2*h)
                dydx_aux = (flowmap_aux[i,j,0,1] - flowmap_aux[i,j,1,1])/(2*h)
                dydy_aux = (flowmap_aux[i,j,2,1] - flowmap_aux[i,j,3,1])/(2*h)
                
                dxdx_main = (flowmap_aux[i+1,j,-1,0] - flowmap_aux[i-1,j,-1,0])/(2*dx)
                dxdy_main = (flowmap_aux[i,j+1,-1,0] - flowmap_aux[i,j-1,-1,0])/(2*dy)
                dydx_main = (flowmap_aux[i+1,j,-1,1] - flowmap_aux[i-1,j,-1,1])/(2*dx)
                dydy_main = (flowmap_aux[i,j+1,-1,1] - flowmap_aux[i,j-1,-1,1])/(2*dy)
                
                off_diagonal_aux = dxdx_aux*dxdy_aux + dydx_aux*dydy_aux
                C_aux = np.array([[dxdx_aux**2 + dydx_aux**2, 
                                   off_diagonal_aux],
                                  [off_diagonal_aux, 
                                   dxdy_aux**2 + dydy_aux**2]])
                
                off_diagonal_main = dxdx_main*dxdy_main + dydx_main*dydy_main
                C_main = np.array([[dxdx_main**2 + dydx_main**2, 
                                    off_diagonal_main],
                                   [off_diagonal_main, 
                                    dxdy_main**2 + dydy_main**2]])
                
                _,evecs_tmp = np.linalg.eigh(C_aux)
                evals_tmp = np.linalg.eigvalsh(C_main)
                eigvals[i,j,:] = evals_tmp
                eigvecs[i,j,:,:] = evecs_tmp
                
    else:
        for i in prange(1,nx-1):
            for j in range(1,ny-1):
                dxdx = (flowmap_aux[i,j,0,0] - flowmap_aux[i,j,1,0])/(2*h)
                dxdy = (flowmap_aux[i,j,2,0] - flowmap_aux[i,j,3,0])/(2*h)
                dydx = (flowmap_aux[i,j,0,1] - flowmap_aux[i,j,1,1])/(2*h)
                dydy = (flowmap_aux[i,j,2,1] - flowmap_aux[i,j,3,1])/(2*h)
               
                off_diagonal = dxdx*dxdy + dydx*dydy
                C = np.array([[dxdx**2 + dydx**2, off_diagonal],
                               [off_diagonal, dxdy**2 + dydy**2]])
                
                evals_tmp,evecs_tmp = np.linalg.eigh(C)
                eigvals[i,j,:] = evals_tmp
                eigvecs[i,j,:,:] = evecs_tmp  
            
    return eigvals,eigvecs


@njit(parallel=True)
def C_eig_2D(flowmap,dx,dy):
    """
    Compute eigenvalues and eigenvectors of Cauchy Green tensor in 2D from flowmap which is
    solution of ode over a grid defined by x,y for integration time T.

    Parameters
    ----------
    flowmap : np.ndarray, shape = (nx,ny,2)
        array containing final positions of initial grid from t0 to t0+T.
    dx : float
        grid spacing in x-direction.
    dy : float
        grid spacing in y-direction.

    Returns
    -------
    eigvals : np.ndarray, shape = (nx,ny,2)
        array containing eigenvalues.
    eigvecs : np.ndarray, shape = (nx,ny,2,2)
        array containing eigenvectors.

    """

    nx,ny = flowmap.shape[:-1]
    eigvals = np.zeros((nx,ny,2),float64)
    eigvecs = np.zeros((nx,ny,2,2),float64)
    for i in prange(1,nx-1):
        for j in range(1,ny-1):
            dxdx = (flowmap[i+1,j,0] - flowmap[i-1,j,0])/(2*dx)
            dxdy = (flowmap[i,j+1,0] - flowmap[i,j-1,0])/(2*dy)
            dydx = (flowmap[i+1,j,1] - flowmap[i-1,j,1])/(2*dx)
            dydy = (flowmap[i,j+1,1] - flowmap[i,j-1,1])/(2*dy)
            
            off_diagonal = dxdx*dxdy + dydx*dydy
            C = np.array([[dxdx**2 + dydx**2, off_diagonal],
                           [off_diagonal, dxdy**2 + dydy**2]])
            
            evals_tmp,evecs_tmp = np.linalg.eigh(C)
            eigvals[i,j,:] = evals_tmp          # largest eigenvalue first
            eigvecs[i,j,:,:] = evecs_tmp   
            
    return eigvals,eigvecs


def ftle_from_eig(eigval_max,T):
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
    ftle[np.nonzero(eigval_max>1)] = np.log(eigval_max[np.nonzero(eigval_max>1)])/(2*abs(T))
    
    return ftle


@njit(parallel=True)
def lavd_grid_2D(flowmap_n,tspan,T,vort_interp,xrav,yrav,period_x=0.0,period_y=0.0):
    """
    Compute LAVD from flowmap_n where flowmap_n contains trajectories computed over gridpoints
    defined by xrav,yrav for an integration time T and trajectories are returned at times given by
    tspan. vort_interp is an interpolant function of vorticity over (at least) that time window.
    
    Parameters
    ----------
    flowmap_n : np.ndarray, shape = (nx,ny,n,2)
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

    Returns
    -------
    lavd : np.ndarray, shape = (nx,ny)
        array containing lavd values.

    """

    nx,ny,n = flowmap_n.shape[:-1]
    npts = len(xrav)
    vort_avg = np.zeros(n,float64)
    for k in prange(n):
        gpts = np.column_stack((tspan[k]*np.ones(npts),xrav,yrav))
        vort_k = vort_interp(gpts)
        vort_avg[k] = np.mean(vort_k)
        
    lavd = np.zeros((nx,ny),float64)
    dt = abs(tspan[1] - tspan[0])
    if period_x + period_y == 0.0:
        tspan = np.expand_dims(tspan,axis=1)
        for i in prange(nx):
            for j in range(ny):
                pts = np.concatenate((tspan,flowmap_n[i,j,:,:]),1)
                vort_traj = vort_interp(pts)
                integrand = np.abs(vort_traj - vort_avg)
                lavd[i,j] = composite_simpsons(integrand,dt)
    elif period_x and period_y:
        for i in prange(nx):
            for j in range(ny):
                pts = np.column_stack((tspan,
                                       flowmap_n[i,j,:,0]%period_x,
                                       flowmap_n[i,j,:,1]%period_y))
                vort_traj = vort_interp(pts)
                integrand = np.abs(vort_traj - vort_avg)
                lavd[i,j] = composite_simpsons(integrand,dt)
    elif period_x:
        for i in prange(nx):
            for j in range(ny):
                pts = np.column_stack((tspan,
                                       flowmap_n[i,j,:,0]%period_x,
                                       flowmap_n[i,j,:,1]))
                vort_traj = vort_interp(pts)
                integrand = np.abs(vort_traj - vort_avg)
                lavd[i,j] = composite_simpsons(integrand,dt)
    elif period_y:
        for i in prange(nx):
            for j in range(ny):
                pts = np.column_stack((tspan,
                                       flowmap_n[i,j,:,0],
                                       flowmap_n[i,j,:,1]%period_y))
                vort_traj = vort_interp(pts)
                integrand = np.abs(vort_traj - vort_avg)
                lavd[i,j] = composite_simpsons(integrand,dt)
            
    return lavd

@njit(parallel=True)
def ftle_grid_ND(flowmap,IC,T,dX):
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
    absT = abs(T)
    for k in prange(npts):
        inds = unravel_index(k,grid_shape)
        dXdir = np.zeros(ndims,numba.int64)
        for ii,ind in enumerate(inds):
            if ind == 0:
                dXdir[ii] = 1
            elif ind == grid_shape[ii]-1:
                dXdir[ii] = -1
            else:
                dXdir[ii] = 0  
        Df = np.zeros((ndims,ndims),float64)                   
        for i in range(ndims):
            for j in range(ndims):
                Df[i,j] = finite_diff_ND(flowmap[:,i],inds,dX[j],j,grid_shape,dXdir[j])
                
        
        C = np.dot(Df.T,Df)
        max_eig = np.linalg.eigvalsh(C)[-1]
        if max_eig > 1:
            ftle[k] = 1/(2*absT)*log(max_eig)
        else: 
            ftle[k] = 0
            
    return ftle
    

@njit(parallel=True)
def ile_2D_func(vel,x,y,t0=None,h=1e-3):
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

    Returns
    -------
    ile : np.ndarray, shape = (nx,ny)
        array containing ile values.

    """

    nx,ny = len(x), len(y)
    ile = np.zeros((nx,ny),float64)
    if t0 is None:
        dx_vec = np.array([h,0.0],float64)
        dy_vec = np.array([0.0,h],float64)
        for i in prange(1,nx-1):
            for j in range(1,ny-1):
                    
                pt = np.array([x[i],y[j]],float64)
                
                dudx,dvdx = (vel(pt + dx_vec) - vel(pt - dx_vec))/(2*h)
                dudy,dvdy = (vel(pt + dy_vec) - vel(pt - dy_vec))/(2*h)
                
                grad_vel = np.array([[dudx,dudy],[dvdx,dvdy]])
                S = 0.5*(grad_vel + grad_vel.T)
                ile[i,j] = np.linalg.eigvalsh(S)[-1]
    else:
        dx_vec = np.array([0.0,h,0.0],float64)
        dy_vec = np.array([0.0,0.0,h],float64)
        for i in prange(1,nx-1):
            for j in range(1,ny-1):
                    
                pt = np.array([t0,x[i],y[j]],float64)
                
                dudx,dvdx = (vel(pt + dx_vec) - vel(pt - dx_vec))/(2*h)
                dudy,dvdy = (vel(pt + dy_vec) - vel(pt - dy_vec))/(2*h)
                
                grad_vel = np.array([[dudx,dudy],[dvdx,dvdy]])
                S = 0.5*(grad_vel + grad_vel.T)
                ile[i,j] = np.linalg.eigvalsh(S)[-1]        
            
    return ile

@njit(parallel=True)
def S_eig_2D_func(vel,x,y,t0=None,h=1e-3):
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
        step size to be used in finite differencing. The default is 1e-2.

    Returns
    -------
    eigvals : np.ndarray, shape = (nx,ny)
        array containg eigenvalues of S.
    eigvecs : np.ndarray, shape = (nx,ny)
        array containing eigenvectors of S.

    """

    nx,ny = len(x), len(y)
    eigvals = np.zeros((nx,ny,2),float64)
    eigvecs = np.zeros((nx,ny,2,2),float64)
    if t0 is None:
        dx_vec = np.array([h,0.0],float64)
        dy_vec = np.array([0.0,h],float64)
        for i in prange(1,nx-1):
            for j in range(1,ny-1):
                pt = np.array([x[i],y[j]],float64)
                
                dudx,dvdx = (vel(pt + dx_vec) - vel(pt - dx_vec))/(2*h)
                dudy,dvdy = (vel(pt + dy_vec) - vel(pt - dy_vec))/(2*h)
                grad_vel = np.array([[dudx,dudy],[dvdx,dvdy]])
                S = 0.5*(grad_vel + grad_vel.T)
                evals_tmp,evecs_tmp = np.linalg.eigh(S)
                eigvals[i,j,:] = evals_tmp
                eigvecs[i,j,:,:] = evecs_tmp
    else:
        dx_vec = np.array([0.0,h,0.0],float64)
        dy_vec = np.array([0.0,0.0,h],float64)
        for i in prange(1,nx-1):
            for j in range(1,ny-1):
                pt = np.array([t0,x[i],y[j]],float64)
                
                dudx,dvdx = (vel(pt + dx_vec) - vel(pt - dx_vec))/(2*h)
                dudy,dvdy = (vel(pt + dy_vec) - vel(pt - dy_vec))/(2*h)
                grad_vel = np.array([[dudx,dudy],[dvdx,dvdy]])
                S = 0.5*(grad_vel + grad_vel.T)
                evals_tmp,evecs_tmp = np.linalg.eigh(S)
                eigvals[i,j,:] = evals_tmp
                eigvecs[i,j,:,:] = evecs_tmp    
                 
            
    return eigvals,eigvecs


@njit(parallel=True)
def S_2D_func(vel,x,y,t0=None,h=1e-3):
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

    Returns
    -------
    S : np.ndarray, shape = (nx,ny)
        S11,S12, and S22 components of S tensor.

    """

    nx,ny = len(x), len(y)
    S = np.zeros((nx,ny,3),float64)
    if t0 is None:
        dx_vec = np.array([h,0.0],float64)
        dy_vec = np.array([0.0,h],float64)
        for i in prange(nx):
            for j in range(ny):
                pt = np.array([x[i],y[j]],float64)
                
                dudx,dvdx = (vel(pt + dx_vec) - vel(pt - dx_vec))/(2*h)
                dudy,dvdy = (vel(pt + dy_vec) - vel(pt - dy_vec))/(2*h)
                S[i,j,:] = np.array([dudx,0.5*(dudy + dvdx),dvdy])
    else:
        dx_vec = np.array([0.0,h,0.0],float64)
        dy_vec = np.array([0.0,0.0,h],float64)
        for i in prange(nx):
            for j in range(ny):
                pt = np.array([t0,x[i],y[j]],float64)
                
                dudx,dvdx = (vel(pt + dx_vec) - vel(pt - dx_vec))/(2*h)
                dudy,dvdy = (vel(pt + dy_vec) - vel(pt - dy_vec))/(2*h)
                S[i,j,:] = np.array([dudx,0.5*(dudy + dvdx),dvdy])        
             
            
    return S

@njit(parallel=True)
def ile_2D_data(u,v,dx,dy):
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

    Returns
    -------
    ile : np.ndarray, shape = (nx,ny)
        array containing ile values.

    """

    nx,ny = u.shape
    ile = np.zeros((nx,ny))
    for i in prange(1,nx-1):
        for j in range(1,ny-1):
                
            dudx = (u[i+1,j] - u[i-1,j])/(2*dx)
            dudy = (u[i,j+1] - u[i,j-1])/(2*dy)
            dvdx = (v[i+1,j] - v[i-1,j])/(2*dx)
            dvdy = (v[i,j+1] - v[i,j-1])/(2*dy)
            grad_vel = np.array([[dudx,dudy],[dvdx,dvdy]])
            S = 0.5*(grad_vel + grad_vel.T)
            ile[i,j] = np.linalg.eigvalsh(S)[-1]
            
    return ile

@njit(parallel=True)
def S_eig_2D_data(u,v,dx,dy):
    """
    Compute eigenvalues and eigenvectors of Eulerian rate of strain tensor in 2D from u,v which
    describe the x,y velocity.

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

    Returns
    -------
    eigvals : np.ndarray, shape = (nx,ny,2)
        array containg eigenvalues of S.
    eigvecs : np.ndarray, shape = (nx,ny,2,2)
        array containing eigenvectors of S.

    """

    nx,ny = u.shape
    eigvals = np.zeros((nx,ny,2),float64)
    eigvecs = np.zeros((nx,ny,2,2),float64)
    for i in prange(1,nx-1):
        for j in range(1,ny-1):

            dudx = (u[i+1,j] - u[i-1,j])/(2*dx)
            dudy = (u[i,j+1] - u[i,j-1])/(2*dy)
            dvdx = (v[i+1,j] - v[i-1,j])/(2*dx)
            dvdy = (v[i,j+1] - v[i,j-1])/(2*dy)
            grad_vel = np.array([[dudx,dudy],[dvdx,dvdy]])
            S = 0.5*(grad_vel + grad_vel.T)
            evals_tmp,evecs_tmp = np.linalg.eigh(S)
            eigvals[i,j,:] = evals_tmp
            eigvecs[i,j,:,:] = evecs_tmp 
 
            
    return eigvals,eigvecs



def ivd_grid_2D(vort,vort_avg):
    """
    Compute IVD at an instant from vort and vort_avg.

    Parameters
    ----------
    vort : np.ndarray, shape = (nx,ny)
        array containg vorticity values.
    vort_avg : float
        value of average spatial vorticity at specific time.

    Returns
    -------
    ivd : np.ndarray, shape = (nx,ny)
        array containing values of ivd.

    """
    
    ivd = np.abs(vort - vort_avg)
    return ivd

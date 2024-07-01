from math import sin, cos, pi, cosh, tanh, sqrt
import numpy as np
from numba import njit, cfunc
from numbalsoda import lsoda_sig
from interpolation.splines import UCGrid, prefilter, eval_spline

def get_interp_arrays_2D(tvals,xvals,yvals,U,V):
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

    nt,nx,ny = U.shape
    grid_vel = UCGrid((tvals[0],tvals[-1],nt),(xvals[0],xvals[-1],nx),(yvals[0],yvals[-1],ny))
    C_eval_u = prefilter(grid_vel,U,out=None,k=3)
    C_eval_v = prefilter(grid_vel,V,out=None,k=3)
    
    return grid_vel, C_eval_u, C_eval_v

def get_flow_2D(grid_vel,C_eval_u,C_eval_v,spherical=0,extrap_mode='constant',r=6371.):
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
        def flow_rhs(t,y,dy,p):
            """
            p[0] = int_direction
            """
            tt = p[0]*t
            xx = ((y[0]-180)%360)-180
            yy = y[1]
            point = np.array([tt,xx,yy])
            dy[0] = (p[0]*eval_spline(grid_vel,C_eval_u,point,out=None,k=3,diff="None",
                                      extrap_mode=extrap_mode))*180/(pi*r*cos(yy*pi/180))
            dy[1] = (p[0]*eval_spline(grid_vel,C_eval_v,point,out=None,k=3,diff="None",
                                      extrap_mode=extrap_mode))*180/(pi*r)
    elif spherical == 2:
        @cfunc(lsoda_sig)
        def flow_rhs(t,y,dy,p):
            """
            p[0] = int_direction
            """
            tt = p[0]*t
            xx = y[0]%360
            yy = y[1]
            point = np.array([tt,xx,yy])
            dy[0] = (p[0]*eval_spline(grid_vel,C_eval_u,point,out=None,k=3,diff="None",
                                      extrap_mode=extrap_mode))*180/(pi*r*cos(yy*pi/180))
            dy[1] = (p[0]*eval_spline(grid_vel,C_eval_v,point,out=None,k=3,diff="None",
                                      extrap_mode=extrap_mode))*180/(pi*r)
    else:
        @cfunc(lsoda_sig)
        def flow_rhs(t,y,dy,p):
            """
            p[0] = int_direction
            """
            tt = p[0]*t
            point = np.array([tt,y[0],y[1]])
            dy[0] = (p[0]*eval_spline(grid_vel,C_eval_u,point,out=None,k=3,diff="None",
                                      extrap_mode=extrap_mode))
            dy[1] = (p[0]*eval_spline(grid_vel,C_eval_v,point,out=None,k=3,diff="None",
                                      extrap_mode=extrap_mode))
        
    funcptr = flow_rhs.address
        
    return funcptr


def get_callable_2D(grid_vel,C_eval_u,C_eval_v,spherical=0,extrap_mode='constant',r=6371.):
    """
    Create a jit-callable spline for the ode defined by the vector field (U,V) defined over
    a spatial grid given by (xvals,yvals) over times defined by tvals.

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
        @njit
        def vel_spline(point):

            ui = eval_spline(grid_vel,C_eval_u,point,out=None,k=3,diff="None",
                                      extrap_mode=extrap_mode)*180/(pi*r*cos(point[2]*pi/180))
            vi = eval_spline(grid_vel,C_eval_v,point,out=None,k=3,diff="None",
                                      extrap_mode=extrap_mode)*180/(pi*r)
            return ui,vi
        
    else:
        @njit
        def vel_spline(point):
            
            ui = eval_spline(grid_vel,C_eval_u,point,out=None,k=3,diff="None",
                             extrap_mode=extrap_mode)
            vi = eval_spline(grid_vel,C_eval_v,point,out=None,k=3,diff="None",
                             extrap_mode=extrap_mode)
            return ui,vi
        
    return vel_spline
        

    
    

def get_flow_linear_2D(grid_vel,U,V,spherical=0,
                return_interp=False,extrap_mode='constant',r=6371.):
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
    return_spline : boolean, optional
        flag to determine if spline is returned. The default is False.
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
        def flow_rhs(t,y,dy,p):
            """
            p[0] = int_direction
            """
            tt = p[0]*t
            xx = ((y[0]-180)%360)-180
            yy = y[1]
            point = np.array([tt,xx,yy])
            dy[0] = (p[0]*eval_spline(grid_vel,U,point,out=None,k=1,diff="None",
                                      extrap_mode=extrap_mode))*180/(pi*r*cos(yy*pi/180))
            dy[1] = (p[0]*eval_spline(grid_vel,V,point,out=None,k=1,diff="None",
                                      extrap_mode=extrap_mode))*180/(pi*r)
    elif spherical == 2:
        @cfunc(lsoda_sig)
        def flow_rhs(t,y,dy,p):
            """
            p[0] = int_direction
            """
            tt = p[0]*t
            xx = y[0]%360
            yy = y[1]
            point = np.array([tt,xx,yy])
            dy[0] = (p[0]*eval_spline(grid_vel,U,point,out=None,k=1,diff="None",
                                      extrap_mode=extrap_mode))*180/(pi*r*cos(yy*pi/180))
            dy[1] = (p[0]*eval_spline(grid_vel,V,point,out=None,k=1,diff="None",
                                      extrap_mode=extrap_mode))*180/(pi*r)
    else:
        @cfunc(lsoda_sig)
        def flow_rhs(t,y,dy,p):
            """
            p[0] = int_direction
            """
            tt = p[0]*t
            point = np.array([tt,y[0],y[1]])
            dy[0] = (p[0]*eval_spline(grid_vel,U,point,out=None,k=1,diff="None",
                                      extrap_mode=extrap_mode))
            dy[1] = (p[0]*eval_spline(grid_vel,V,point,out=None,k=1,diff="None",
                                      extrap_mode=extrap_mode))
        
    funcptr = flow_rhs.address 
    
    return funcptr


def get_callable_linear_2D(grid_vel,U,V,spherical=0,extrap_mode='constant',r=6371.):
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

    Returns
    -------
    vel_spline : jit-callable
        jit-callable function for vector field.

    """
    
    if spherical == 1:
        @njit
        def vel_spline(point):

            ui = eval_spline(grid_vel,U,point,out=None,k=1,diff="None",
                                      extrap_mode=extrap_mode)*180/(pi*r*cos(point[2]*pi/180))
            vi = eval_spline(grid_vel,V,point,out=None,k=1,diff="None",
                                      extrap_mode=extrap_mode)*180/(pi*r)
            return ui,vi
        
    else:
        @njit
        def vel_spline(point):
            
            ui = eval_spline(grid_vel,U,point,out=None,k=1,diff="None",
                             extrap_mode=extrap_mode)
            vi = eval_spline(grid_vel,V,point,out=None,k=1,diff="None",
                             extrap_mode=extrap_mode)
            return ui,vi
        
    return vel_spline


def get_predefined_flow(flow_str,int_direction=1.,return_default_params=True,
                        return_domain=True):
    """
    Create a C callback for one of the predefined flows.

    Parameters
    ----------
    flow_str : str
        string representing which flow to retrieve. Currently 'double_gyre'
        and 'abc' are supported.
    int_direction : float, optional
        direction of integration (either -1. or 1.). The default is 1.
    return_default_params : boolean, optional
        flag to determine if default parameters will be returned. The default is True.
    return_domain : boolean, optional
        flag to determine if domain will be returned. The default is True.

    Returns
    -------
    funcptr : int
        address of C callback.
    default_params : np.ndarray, shape = (nprms,), optional
        default parameters.
    domain : tuple, optional
        array containing endpoints of domain for each dimension.
    

    """
    
    match flow_str:
        case "double_gyre":
            @cfunc(lsoda_sig)
            def _double_gyre(t,y,dy,p):
                """
                p[0] = int_direction, p[1] = A, p[2] = eps, p[3] = alpha, p[4] = omega,
                p[5] = psi, p[6] = eta
                """
                tt = p[0]*t
                a = p[2]*sin(p[4]*tt + p[5])
                b = 1 - 2*p[2]*sin(p[4]*tt + p[5])
                f = a*y[0]**2 + b*y[0]
                df = 2*a*y[0] + b
                dy[0] = p[0]*(-pi*p[1]*sin(pi*f)*cos(pi*y[1]) - p[3]*y[0] + p[6])
                dy[1] = p[0]*(pi*p[1]*cos(pi*f)*sin(pi*y[1])*df - p[3]*y[1] + p[6])
            
            funcptr = _double_gyre.address
                
            if return_default_params:
                A = 0.1
                eps = 0.1
                alpha = 0.
                omega = 0.2*pi
                psi = 0.
                eta = 0.
                
                default_params = np.array([int_direction,A,eps,alpha,omega,psi,eta])
                
            if return_domain:
                domain = ((0.,2.),(0.,1.))
                
            
        case 'abc':
            @cfunc(lsoda_sig)
            def _abc(t,y,dy,p):
                """
                p[0] = int_direction, p[1] = A-amplitude, p[2] = B-amplitude, p[3] = C-amplitude
                p[4] = forcing amplitdue
                """
                tt = p[0]*t
                dy[0] = p[0]*((p[1]+p[4]*sin(pi*tt))*sin(y[2]) + p[3]*cos(y[1]))
                dy[1] = p[0]*(p[2]*sin(y[0]) + (p[1] + p[4]*sin(pi*tt))*cos(y[1]))
                dy[2] = p[0]*(p[3]*sin(y[1]) + p[2]*cos(y[1]))
                
            if return_default_params:
                A = 3**0.5
                B = 2**0.5
                C = 1.
                f = 0.5
                
                
                default_params = np.array([int_direction,A,B,C,f])
                
            if return_domain:
                domain = ((0.,2*pi),(0.,2*pi),(0.,2*pi))
                
    match [return_default_params,return_domain]:
        case [False,False]:
            return funcptr
        case [True,False]:
            return funcptr, default_params
        case [False,True]:
            return funcptr, domain
        case [True,True]:
            return funcptr, default_params, domain                
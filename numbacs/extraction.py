import numpy as np
from numba import njit, prange
import numba                                 
from utils import composite_simpsons_38_irregular, unravel_index, dist_2d, ravel_index
from interpolation.splines import UCGrid, eval_spline, prefilter
from math import copysign, floor, acos, atan2, pi
from scipy.ndimage import label, generate_binary_structure

parallel_flag=True

@njit
def _reorient_eigvec(pt,x,y,eigvec):
    """
    Interpolates eigvec at point pt, where eigvec is defined over np.meshgrid(x,y),
    with possible orientation discontinuity

    Parameters
    ----------
    pt : np.ndarray, shape = (2,)
        x,y position at which eigvec will be interpolated.
    x : np.ndarray, shape = (nx,)
        array containing x-values.
    y : np.ndarray, shape = (ny,)
        array containing y-values.
    eigvec : np.ndarray, shape = (nx,ny,2)
        array containing eigenvectors.

    Returns
    -------
    eigvec_interp : np.ndarray, shape = (2,)
        interpolated eigenvector at pt.

    """

    # Find which grid box point lies in
    indx = np.searchsorted(x, pt[0])
    indy = np.searchsorted(y, pt[1])
    inds = np.array([[indx-1,indy-1],[indx-1,indy],[indx,indy-1],[indx,indy]],numba.int32)
    grid_eigvecs = np.zeros((4,2),numba.float64)
    grid_eigvec_x = np.zeros((2,2),numba.float64)
    grid_eigvec_y = np.zeros((2,2),numba.float64)
    
    # Orient eigenvectors
    for k in range(4):
        grid_eigvecs[k,:] = eigvec[inds[k,0],inds[k,1],:]
        if k > 0:
            grid_eigvecs[k,:] = copysign(1,np.dot(grid_eigvecs[k-1,:],
                                                  grid_eigvecs[k,:]))*grid_eigvecs[k,:]
        i,j = divmod(k,2)    
        grid_eigvec_x[i,j] = grid_eigvecs[k,0]
        grid_eigvec_y[i,j] = grid_eigvecs[k,1]             


    grid_eigvecs = grid_eigvecs.reshape(2,2,2)
    grid = ((x[indx-1],x[indx],2),(y[indy-1],y[indy],2))
    
    # Linear interpolant for eigenvectors based on grid box
    eigvec_interp = np.array([eval_spline(grid,grid_eigvec_x,pt,k=1,
                                          diff="None",extrap_mode="constant"),
                              eval_spline(grid,grid_eigvec_y,pt,k=1,
                                          diff="None",extrap_mode="constant")])
    return  eigvec_interp

       

@njit
def _alpha(pt,eigval_max):
    """
    Compute alpha to be used in tensorline computation at point pt using eigval_max which is an 
    interpolant function

    Parameters
    ----------
    pt : np.ndarray, shape = (2,)
        x,y position at which alpha will be interpolated.
    eigval_max : jit-callable
        interpolant function for maximum eigenvalue.

    Returns
    -------
    a : float
        interpolated alpha value.

    """

    a = ((eigval_max(pt) - 1/eigval_max(pt))/(eigval_max(pt) + 1/eigval_max(pt)))**2
    return a

@njit
def _in_domain(pt,domain):
    """
    Check if point pt is in domain where domain is rectangular region.

    Parameters
    ----------
    pt : np.ndarray, shape = (2,)
        x,y position at which will be checked.
    domain : np.ndarray, shape = ((2,2))
        array defining rectangular domain [[xmin,xmax],[ymin,ymax]].

    Returns
    -------
    bool
        truth value determining if pt is in domain or not.

    """

    
    if pt[0] <= domain[0,0] or pt[0] >= domain[0,1] or pt[1] <= domain[1,0] or pt[1] >= domain[1,1]:
        return False
    else:
        return True
    
@njit
def _in_region(pt,xvals,yvals,arr):
    """
    Determines if pt is in a region defined over a meshgrid defined by xvals,yvals where arr is a
    boolean array which defines the region.

    Parameters
    ----------
    pt : np.ndarray, shape = (2,)
        x,y position at which will be checked.
    xvals : np.ndarray, shape = (nx,)
        array containing x-values.
    yvals : np.ndarray, shape = (ny,)
        array containing y-values.
    arr : np.ndarray, shape = (nx,ny)
        array defining the region.

    Returns
    -------
    near : bool
        truth value determining if pt is in region.

    """

    indx1 = np.searchsorted(xvals, pt[0])
    indy1 = np.searchsorted(yvals, pt[1])
    inds = np.array([[indx1-1,indy1-1],[indx1-1,indy1],[indx1,indy1-1],[indx1,indy1]],numba.int32)
    for k in range(4):
        near = arr[inds[k,0],inds[k,1]]
        if near == 1:
            break
     
    return near
    
@njit
def rk4_tensorlines(eigval_max,eigvec_min,xvals,yvals,ic_ind,h,steps,U0,lf):
    """
    Compute tensorlines in eigvec_min field originially defined on over xvals,yvals.

    Parameters
    ----------
    eigval_max : jit-callable
        interpolant function of maximum eigenvalue.
    eigvec_min : np.ndarray, shape = (nx,ny,2)
        array of maximum eigenvectors.
    xvals : np.ndarray, shape = (nx,)
        array containing x-values.
    yvals : np.ndarray, shape = (ny,)
        array containing y-values.
    ic_ind : np.ndarray, shape = (2,)
        array containing xind,yind corresponding to initial condition.
    h : float
        step size used in the rk4 solver.
    steps : int
        maximum number of steps allowed for rk4 solver.
    U0 : np.ndarray, shape = (nx,ny)
        array defining region used by _in_region which satisfies LCS criteria.
    lf : float
        maximum failure distance of LCS criteria allowed.

    Returns
    -------
    tensorline : np.ndarray, shape = (len(d1lcs) + len(d2lcs),2)
        array containing tensorline representing candidate LCS.

    """

    domain = np.array([[xvals[0],yvals[0]],[xvals[-1],yvals[-1]]])    # for _in_domain
    for direction in np.array([1,-1],numba.int32):
        init_direction = direction                  # initialize integration direction
        L=0                                         # initialize failure length
        y = np.zeros((steps+1,2),numba.float64)
        y[0,:] = np.array([xvals[ic_ind[0]],yvals[ic_ind[1]]],numba.float64) # set initial condition
        init_vec = eigvec_min[ic_ind[0],ic_ind[1],:]    # set initial vector
        a1 = _alpha(y[0,:],eigval_max)              # initial alpha
        i = 0
        steps_left = steps                    # used to make sure don't get stuck in infinite loop
        error_flag = 0                        # used to end earlier if stuck in loop
        remove_inds = 0
        while L<lf and i<steps:
            # rk4 to integrate tensorlines from eigvec_min field
            k1 = init_vec*a1
            if init_direction == -1:
                k1 = -k1
            yk2 = y[i,:]+0.5*h*k1
            a2 = _alpha(yk2,eigval_max)
            k2 = _reorient_eigvec(yk2,xvals,yvals,eigvec_min)*a2   # reorient and scale eigenvector
            if np.dot(k1,k2) < 0:                           # make sure direction is continuous
                k2 = -k2
            yk3 = y[i,:]+0.5*h*k2
            a3 = _alpha(yk3,eigval_max)
            k3 = _reorient_eigvec(yk3,xvals,yvals,eigvec_min)*a3
            if np.dot(k2,k3) < 0:
                k3 = -k3
            yk4 = y[i,:]+h*k3
            a4 = _alpha(yk4,eigval_max)
            k4 = _reorient_eigvec(yk4,xvals,yvals,eigvec_min)*a4
            if np.dot(k3,k4) < 0:
                k4 = -k4
            y[i+1,:] = y[i,:] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)   # rk4 step
            
            # check if point is in domain and not stuck on boundary or fixed point
            if _in_domain(y[i+1,:],domain) == False or np.all(np.isclose(y[i,:],y[i+1,:])):
                break
            
            if i > 1:
                r0 = y[i,:] - y[i-1,:]
                r1 = y[i+1,:] - y[i,:]
                if np.dot(r0,r1) >= 0:
                    # set updated init_vector and init_direction
                    init_vec = _reorient_eigvec(y[i+1,:],xvals,yvals,eigvec_min)
                    init_direction = np.sign(np.dot(k4,init_vec))
                    
                    error_flag = 0
                    a1 = _alpha(y[i+1,:],eigval_max)
                    
                    # check if point is in U0 region meeting necessary LCS criteria
                    if _in_region(y[i+1,:],xvals,yvals,U0):
                        L = 0.
                        remove_inds = 0
                    else:
                        L += dist_2d(y[i+1,:],y[i,:])
                        remove_inds += 1
                    i+=1
                else:
                    break
            else:
                # set updated init_vector and init_direction
                init_vec = _reorient_eigvec(y[i+1,:],xvals,yvals,eigvec_min)
                init_direction = np.sign(np.dot(k4,init_vec))
                
                a1 = _alpha(y[i+1,:],eigval_max)
                if _in_region(y[i+1,:],xvals,yvals,U0):
                    L = 0.
                    remove_inds = 0
                else:
                    L = L + dist_2d(y[i+1,:],y[i,:])
                    remove_inds += 1
                i+=1
            
            # break if we're stuck in a loop going back and forth between points or on single point
            steps_left-=1
            if steps_left <= 0 or error_flag > 1:
                break
        
        i -= remove_inds
        if direction == 1:            
            d1lcs = np.flipud(y[:i+1,:])    # direction 1 candidite lcs
        else:
            d2lcs = y[1:i+1,:]              # direction -1 candidate lcs
        tensorline = np.concatenate((d1lcs,d2lcs),axis=0) # concatenate both canditate lcs   
    return tensorline   



@njit(parallel=parallel_flag)
def _arclength_arr(pts):
    """
    Compute arclength of each segment of curve defined by pts.

    Parameters
    ----------
    pts : np.ndarray, shape=(npts,)
        points representing curve for which arclength is to be computed.

    Returns
    -------
    arclength : np.ndarray, shape = (npts-1,)
        array containing arclength of each segment of curve defined by pts.

    """
    arclength = np.zeros(pts.shape[0]-1,numba.float64)
    for k in prange(len(pts)-1):
        arclength[k] = dist_2d(pts[k,:],pts[k+1,:])
        
    return arclength


@njit
def _pts_in_dist_ind(pt,arr,tol,attach_ind=-1):    
    """
    Find all points in arr which are within a distance tolerance of pt.
    
    Parameters
    ----------
    pt : np.ndarray, shape = (2,)
        x,y position of reference point.
    arr : np.ndarray, shape = (npts,3)
        array containing indices and points to check distance from pt.
    tol : float
        distance tolerance.
    attach_ind : int, optional
        index to attach if user specifies. The default is -1.
    
    Returns
    -------
    np.ndarray, shape = (ninds,)
        indices near pt from arr.
    
    """
    inds = np.zeros((arr.shape[0]+1,2),numba.int32)
    for k in range(arr.shape[0]):
        if dist_2d(pt, arr[k,1:]) < tol:
            inds[k,:] = np.array([int(arr[k,0]),1],numba.int32)
    if attach_ind != -1:
        inds[k+1,:] = np.array([attach_ind,1],numba.int32)
        
    return np.unique(inds[inds[:,1]==1,0])   


@njit
def hyp_lcs(tensorlines,lambda_avg,vlines,hlines,dist_tol=1e-1):
    """
    Compute hyperbolic LCS tensorlines from 'rk4_tensorlines', compare nearby tensorlines
    to see which candidatelcs curves are the most attracting/repelling.

    Parameters
    ----------
    tensorlines : np.ndarray, shape = (max_tensorline_len+1,2*nt)
        array containing tensorlines.
    lambda_avg : np.ndarray, shape = (nt,)
        array containing the average attraction/repulsion rate for each tensorline.
    vlines : np.ndarray, shape = (nlines,)
        array containing values of vertical lines.
    hlines : np.ndarray, shape = (nlines,)
        array containing values of horizontal lines.
    dist_tol : float, optional
        distance tolerance used to identify nearby tensorlines. The default is 1e-1.

    Returns
    -------
    np.ndarray, shape = (ii,)
        indices corresponding to most attracting/repelling LCS.

    """

    
    nt = int(tensorlines.shape[-1]/2)       # number of tensorlines
    max_intersects=10                       # maximum times we look at tensorline intersection
    keep_inds = -1*np.ones(nt,numba.int32)    # prealocate array for inds to keep
    jj = 0
    for vl in vlines:
        v_intersect = np.zeros((nt*max_intersects,3),numba.float64) # prealocate 
        ii=0        # initialize counter for storing intersection points
        for k in range(nt): 
            len_ind = int(tensorlines[-1,2*k])   # ind len of tensorline k
            tensorline_tmp = tensorlines[:len_ind,2*k:2*k+2]    # tmp tensorline
            vcross = tensorline_tmp[:,0]>vl     # boolean finding crossings of vline
            vcross_ind = np.argwhere(~vcross == np.roll(vcross,-1))     # find inds of crossings
            if vcross_ind.size == 0:
                continue                        # continue if no crosses
            if vcross_ind[-1] == len_ind-1:
                vcross_ind = vcross_ind[:-1]    # remove last if endpoint
            for i,ind in enumerate(vcross_ind):
                ind = ind[0]    # need to squeeze array
                # find which index is closest to vline
                vcross_ind_tmp = ind + np.argmin(np.array([abs(tensorline_tmp[ind,1]-vl), 
                                                          abs(tensorline_tmp[ind+1,1]-vl)]))  
                
                # store intersection point of kth tensorline and index
                v_intersect[ii+i,:] = np.array([k,tensorline_tmp[vcross_ind_tmp,0],
                                                  tensorline_tmp[vcross_ind_tmp,1]])
            ii += i+1   # increment counter by number of intersections of kth tensorline with vl
        
        # iterate through intersection points of hl
        for k in range(len(v_intersect[:ii,0])):
            t_ind = int(v_intersect[k,0])   # tensorline index corresponding to intersection             
            ot_inds = v_intersect[:,0] != t_ind     # find all other tensorlines intersecting hl
            # find which of ot_inds are within dist_tol of t_ind
            nearby_inds = _pts_in_dist_ind(v_intersect[k,1:],v_intersect[ot_inds,:],
                                       dist_tol,attach_ind=t_ind)
            
            # if only t_ind, continue
            if nearby_inds.size < 2:
                if t_ind not in keep_inds:
                    keep_inds[jj] = t_ind
                    jj+=1
                continue
            max_ind = nearby_inds[np.argmax(lambda_avg[nearby_inds])] # max l_avg from nearby_inds
            
            # if we are already keeping max_inds, continue so we don't double count
            if max_ind in keep_inds:
                continue
            # print(max_ind)
            keep_inds[jj] = max_ind                         # keep the max
            jj+=1
    for hl in hlines:
        h_intersect = np.zeros((nt*max_intersects,3),numba.float64) # prealocate 
        ii = 0      # initialize counter for storing intersection points
        for k in range(nt):
            len_ind = int(tensorlines[-1,2*k])  # ind len of tensorline k
            tensorline_tmp = tensorlines[:len_ind,2*k:2*k+2]    # tmp tensorline
            hcross = tensorline_tmp[:,1]>hl     # boolean finding crossings of hline
            hcross_ind = np.argwhere(~hcross == np.roll(hcross,-1)) # find inds of crossings
            if hcross_ind.size == 0:
                continue                        # continue if no crosses
            if hcross_ind[-1] == len_ind-1:
                hcross_ind = hcross_ind[:-1]    # remove last if endpoint
            for i,ind in enumerate(hcross_ind):
                ind = ind[0]                    # need to squeeze array
                # find which index is closest to hline
                hcross_ind_tmp = ind + np.argmin(np.array([abs(tensorline_tmp[ind,1]-hl), 
                                                          abs(tensorline_tmp[ind+1,1]-hl)]))
                
                # store intersection point of kth tensorline and index
                h_intersect[ii+i,:] = np.array([k,tensorline_tmp[hcross_ind_tmp,0],
                                                  tensorline_tmp[hcross_ind_tmp,1]])
            ii += i+1   # increment counter by number of intersections of kth tensorline with hl
        
        # iterate through intersection points of hl
        for k in range(len(h_intersect[:ii,0])):
            t_ind = int(h_intersect[k,0])   # tensorline index corresponding to intersection
            ot_inds = h_intersect[:,0] != t_ind     # find all other tensorlines intersecting hl
            
            # find which of ot_inds are within dist_tol of t_ind
            nearby_inds = _pts_in_dist_ind(h_intersect[k,1:],h_intersect[ot_inds,:],
                                       dist_tol,attach_ind=t_ind)
            
            # if only t_ind, continue
            if nearby_inds.size < 2:
                if t_ind not in keep_inds:
                    keep_inds[jj] = t_ind
                    jj+=1
                continue
            max_ind = nearby_inds[np.argmax(lambda_avg[nearby_inds])] # max l_avg from nearby_inds
    
            # if we are already keeping max_inds, continue so we don't double count
            if max_ind in keep_inds:
                continue
            keep_inds[jj] = max_ind                         # keep the max
            jj+=1

    return keep_inds[:jj]


@njit
def max_in_radius(arr,r,dx,dy,n=-1):
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
        while np.max(arr) > 0:
            max_ind = np.argmax(arr)
            max_inds[k,:] = unravel_index(max_ind,arr_shape)
            max_vals[k] = arr[max_inds[k,0],max_inds[k,1]]
            max_vals[k] = arr[max_inds[k,0],max_inds[k,1]]
            arr[max(0,max_inds[k,0]-ix):min(nx,max_inds[k,0]+ix),
                max(0,max_inds[k,1]-iy):min(ny,max_inds[k,1]+iy)] = 0
            k+=1
        
        return max_vals[:k], max_inds[:k,:]
    
    else:
        max_inds = np.zeros((n,2),numba.int32)
        max_vals = np.zeros(n,numba.float64)
        for k in range(n):
            max_ind = np.argmax(arr)
            max_inds[k,:] = unravel_index(max_ind,arr_shape)
            max_vals[k] = arr[max_inds[k,0],max_inds[k,1]]
            arr[max(0,max_inds[k,0]-ix):min(nx,max_inds[k,0]+ix),
                max(0,max_inds[k,1]-iy):min(ny,max_inds[k,1]+iy)] = 0
            
        return max_vals, max_inds


@njit(parallel=parallel_flag)
def lcs_region(eigval_max,eigvec_max,dx,dy):
    """
    Compute LCS region.

    Parameters
    ----------
    eigval_max : np.ndarray, shape = (nx,ny)
        array containing maximum eigenvalues.
    eigvec_max : np.ndarray, shape = (nx,ny,2)
        array containing maximum eigenvectors.
    dx : float
        gird spacing in x-direction.
    dy : float
        grid spacing in y-direction.

    Returns
    -------
    lcs_reg : np.ndarray, shape = (nx,ny)
        0,1 array which states if grid point satisfies LCS condition.

    """

    nx,ny = eigval_max.shape
    lcs_reg = np.zeros((nx,ny),numba.int32)
    for i in prange(2,nx-2):
        for j in range(2,ny-2):
            if eigval_max[i,j] <= 1:
                continue
                    
            fxx = (eigval_max[i+1,j] - 2*eigval_max[i,j] + eigval_max[i-1,j])/(dx**2)
            fyy = (eigval_max[i,j+1] - 2*eigval_max[i,j] + eigval_max[i,j-1])/(dy**2)
            fxy = (eigval_max[i+1,j+1] - eigval_max[i+1,j-1] -
                   eigval_max[i-1,j+1] + eigval_max[i-1,j-1])/(4*dx*dy)
            
            eigvec_max0 = eigvec_max[i,j,:]
            ex,ey = eigvec_max0
            
            region_criteria = ex*(fxx*ex + fxy*ey) + ey*(fxy*ex + fyy*ey)
            
            if region_criteria <= 0:
                lcs_reg[i,j] = 1
            
            
    return lcs_reg


def _compute_lcs(eigval_max,eigvecs,x,y,h,steps,lf,lmin,r,nmax):
    """
    Compute hyperbolic LCS using eigval_max and eigvecs obtained from Cauchy Green
    tensor.

    Parameters
    ----------
    eigval_max : np.ndarray, shape = (nx,ny)
        array containing maximum eigenvalues.
    eigvecs : np.ndarray, shape = (nx,ny,2,2)
        array containing eigenvectors.
    x : np.ndarray, shape = (nx,)
        array containing x-values.
    y : np.ndarray, shape = (ny,)
        array containing y-values.
    h : float
        step size used in Runge-Kutta solver.
    steps : int
        maximum number of steps for Runge-Kutta solver.
    lf : float
        maximum failure distance of LCS criteria allowed.
    lmin : float
        minumum allowed length for LCS.
    r : float
        radius in which points will be discared after a maximum is found at the center.
    nmax : int
        number of maxima, i.e. initial conditions for tensorlines.

    Returns
    -------
    lcs_arr : np.ndarray, shape = (2*steps,2*k)
        array containing candidate LCS tensorlines.
    lambda_avg : np.ndarray, shape = (k,)
        average attraction/repullsion rate of each candidate LCS.
    arclength : np.ndarray, shape = (k,)
        arclength of each candidate LCS.

    """


    nx,ny = len(x),len(y)
    dx,dy = abs(x[1] - x[0]), abs(y[1] - y[0])
    eigval_max_nb = eigval_max[1:-1,1:-1]
    
    max_vals,max_inds = max_in_radius(eigval_max.copy(),r,dx,dy,n=nmax)
    
    grid = UCGrid((x[1],x[-2],nx-2),(y[1],y[-2],ny-2))
    C_eval = prefilter(grid,eigval_max_nb,out=None,k=3)
    @njit
    def eigval_max_spline(point):
        return eval_spline(grid,C_eval,point,out=None,k=3,diff="None",extrap_mode="linear")
    
    U0 = lcs_region(eigval_max,eigvecs[:,:,:,1],dx,dy)
    lcs_arr = np.zeros((2*steps+1,2*len(max_inds)),np.float64)
    lambda_avg = np.zeros(len(max_inds),np.float64)
    arclength = np.zeros(len(max_inds),np.float64)
    k = 0
    for ic_ind in max_inds:
        clcs = rk4_tensorlines(eigval_max_spline,eigvecs[:,:,:,0],x,y,ic_ind,h,steps,U0,lf)
        arclength_arr = _arclength_arr(clcs)
        arclength_tmp = arclength_arr.sum()
        clcs_lambda = eigval_max_spline(clcs)
        if arclength_tmp >= lmin:
            lambda_avg[k] = composite_simpsons_38_irregular(clcs_lambda,arclength_arr)/arclength_tmp
            llen = len(clcs)
            lcs_arr[:llen,2*k:2*k+2] = clcs
            lcs_arr[-1,2*k] = llen
            arclength[k] = arclength_tmp
            k+=1
        del clcs
    lambda_avg = lambda_avg[:k]
    lcs_arr = lcs_arr[:,:2*k]
        
    return lcs_arr, lambda_avg, arclength
        
def compute_lcs(eigval_max,eigvecs,x,y,h,steps,lf,lmin,r,nmax,dtol,nlines):
    """
    Wrapper for _compute_lcs which also performs comparison of close enough LCS
    and returns most attracting/repelling LCS in a list.

    Parameters
    ----------
    eigval_max : np.ndarray, shape = (nx,ny)
        array containing maximum eigenvalues.
    eigvecs : np.ndarray, shape = (nx,ny,2,2)
        array containing eigenvectors.
    x : np.ndarray, shape = (nx,)
        array containing x-values.
    y : np.ndarray, shape = (ny,)
        array containing y-values.
    h : float
        step size used in Runge-Kutta solver.
    steps : int
        maximum number of steps for Runge-Kutta solver.
    lf : float
        maximum failure distance of LCS criteria allowed.
    lmin : float
        minumum allowed length for LCS.
    r : float
        radius in which points will be discared after a maximum is found at the center.
    nmax : int
        number of maxima, i.e. initial conditions for tensorlines.
    dtol : float
        distance tolerance used when comparing candidate LCS.
    nlines : int
        number of lines in each direction used when comparing candidate LCS.

    Returns
    -------
    lcs : list
        collection of most attracting/repelling LCS.

    """
    lcs = []
    lcs_arr_, lambda_avg_, arclength_ = _compute_lcs(eigval_max,eigvecs,x,y,h,steps,lf,lmin,r,nmax)
    lcs_keep_inds = hyp_lcs(lcs_arr_,lambda_avg_,y[0,1:-2:nlines],x[1:-2:nlines,0],dtol)
    for ind in lcs_keep_inds:
        lcs_ilen = int(lcs_arr_[-1,2*ind])
        lcs.append(lcs_arr_[:lcs_ilen,2*ind:2*ind+2])
        
    return lcs


@njit(parallel=parallel_flag)
def ftle_ridge_pts(f,eigvec_max,x,y,sdd_thresh=0.,percentile=0):
    """
    Compute FTLE ridge points by finding points (with subpixel accuracy) at which:
        ftle > 0 (or percentile of ftle),
        directional derivaitve of ftle (in eigvec_max direction) is 0,
        second directional derivative of ftle (in eigvec_max direction) is less than sdd_thresh.

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
    
    nx,ny = eigvec_max.shape[:-1]
    ridge_bool = np.zeros(nx*ny,numba.bool_)
    r_pts = np.zeros((nx*ny,2),numba.float64)
    
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    shape_arr = np.array([nx,ny],numba.int32)
    if percentile == 0:
        f_min = 0.
    else:
        f_min = np.percentile(f, percentile)
    for i in prange(2,nx-2):
        for j in range(2,ny-2):
            pt = np.array([x[i],y[j]])
            f0 = f[i,j]
            if f0 > f_min:
                fx = (f[i+1,j] - f[i-1,j])/(2*dx)
                fy = (f[i,j+1] - f[i,j-1])/(2*dy)
                fxx = (f[i+1,j] - 2*f[i,j] + f[i-1,j])/(dx**2)
                fyy = (f[i,j+1] - 2*f[i,j] + f[i,j-1])/(dy**2)
                fxy = (f[i+1,j+1] - f[i+1,j-1] -
                       f[i-1,j+1] + f[i-1,j-1])/(4*dx*dy)
                
                eigvec_max0 = eigvec_max[i,j,:]
                ex,ey = eigvec_max0
                
                c2 = ex*(fxx*ex + fxy*ey) + ey*(fxy*ex + fyy*ey)
                if c2 < -sdd_thresh:
                    t = -(fx*ex + fy*ey)/c2
                    if abs(t*ex) <= dx/2 and abs(t*ey) <= dy/2:
                        k = ravel_index(np.array([i,j],numba.int32),shape_arr)
                        r_pts[k,:] = pt + np.array([t*ex,t*ey])
                        ridge_bool[k] = True
                        
    return r_pts[ridge_bool,:]


@njit(parallel=parallel_flag)
def _ftle_ridges(f,eigvec_max,x,y,sdd_thresh=0.,percentile=0):
    """
    Compute FTLE ridge points by finding points (with subpixel accuracy) at which:
        ftle > 0 (or percentile of ftle),
        directional derivaitve of ftle (in eigvec_max direction) is 0,
        second directional derivative of ftle (in eigvec_max direction) is less than sdd_thresh.

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
    
    nx,ny = eigvec_max.shape[:-1]
    ridge_bool = np.zeros((nx,ny),numba.bool_)
    r_pts = np.zeros((nx*ny,2),numba.float64)
    
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    shape_arr = np.array([nx,ny],numba.int32)
    if percentile == 0:
        f_min = 0.
    else:
        f_min = np.percentile(f, percentile)
    for i in prange(2,nx-2):
        for j in range(2,ny-2):
            pt = np.array([x[i],y[j]])
            f0 = f[i,j]
            if f0 > f_min:
                fx = (f[i+1,j] - f[i-1,j])/(2*dx)
                fy = (f[i,j+1] - f[i,j-1])/(2*dy)
                fxx = (f[i+1,j] - 2*f[i,j] + f[i-1,j])/(dx**2)
                fyy = (f[i,j+1] - 2*f[i,j] + f[i,j-1])/(dy**2)
                fxy = (f[i+1,j+1] - f[i+1,j-1] -
                       f[i-1,j+1] + f[i-1,j-1])/(4*dx*dy)
                
                eigvec_max0 = eigvec_max[i,j,:]
                ex,ey = eigvec_max0
                
                c2 = ex*(fxx*ex + fxy*ey) + ey*(fxy*ex + fyy*ey)
                if c2 < -sdd_thresh:
                    t = -(fx*ex + fy*ey)/c2
                    if abs(t*ex) <= dx/2 and abs(t*ey) <= dy/2:
                        k = ravel_index(np.array([i,j],numba.int32),shape_arr)
                        r_pts[k,:] = pt + np.array([t*ex,t*ey])
                        ridge_bool[i,j] = True
                        
    return r_pts, ridge_bool, nx, ny


@njit
def _get_ridges(ridge_pts,inds,nx,ny,min_ridge_len):
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
    min_ridge_len : int
        minimum points allowed in ridge.

    Returns
    -------
    np.ndarray, shape = (len(inds),2)
        array containing ridge points for current ridge.

    """
    
    if  len(inds) >= min_ridge_len:
        return ridge_pts[inds,:]

            
def ftle_ridges(f,eigvec_max,x,y,sdd_thresh=0.,percentile=0,min_ridge_len=3):
    """
    From ridge points from _ftle_ridges, extract connected ridges where
    a connected ridge is defined by having continuous neighbors that
    are ridge points.

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
    min_ridge_len : int
        minimum points allowed in ridge. The default is 3.

    Returns
    -------
    list
        list containing each connected ridge.

    """
    
    r_pts_, ridge_bool_, nx, ny = _ftle_ridges(f,eigvec_max,x,y,
                                                sdd_thresh=sdd_thresh,
                                                percentile=percentile)
    nx,ny = f.shape
    s = generate_binary_structure(2,2)
    labels,nlabels = label(ridge_bool_,structure=s)
    ind_arr = np.arange(0,nx*ny).reshape(nx,ny)
    ridges = []
    for i in range(nlabels):
        inds = ind_arr[labels==i]
        ridges.append(_get_ridges(i,r_pts_,inds,labels,nlabels,nx,ny,min_ridge_len))
        
    return [r for r in ridges if r is not None]


@njit(parallel=True)
def _ftle_ridge_pts_connect(eigval_max,eigvec_max,x,y,sdd_thresh=0.,percentile=0):
    """
    Compute FTLE ridge points by finding points (with subpixel accuracy) at which:
        ftle > 0 (or percentile of ftle),
        directional derivaitve of ftle (in eigvec_max direction) is 0,
        second directional derivative of ftle (in eigvec_max direction) is less than sdd_thresh.
        
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
    
    nx,ny = eigval_max.shape
    r_pts = -1*np.ones((nx*ny,3),numba.float64)
    r_vec = np.zeros((nx*ny,2),numba.float64)
    sdd = np.zeros((nx*ny,),numba.float64)
    
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    h = min(dx,dy)
    # set min value allowed for eigval_max
    if percentile == 0:
        eigval_min_val = 1
    else:
        eigval_min_val = np.percentile(eigval_max, percentile)
        
    for i in prange(2,nx-2):
        for j in range(2,ny-2):
            pt = np.array([x[i],y[j]])
            eigval_max0 = eigval_max[i,j]
            # compute derivatives if eigval large enough
            if eigval_max0 > eigval_min_val:
                fx = (eigval_max[i+1,j] - eigval_max[i-1,j])/(2*dx)
                fy = (eigval_max[i,j+1] - eigval_max[i,j-1])/(2*dy)
                fxx = (eigval_max[i+1,j] - 2*eigval_max[i,j] + eigval_max[i-1,j])/(dx**2)
                fyy = (eigval_max[i,j+1] - 2*eigval_max[i,j] + eigval_max[i,j-1])/(dy**2)
                fxy = (eigval_max[i+1,j+1] - eigval_max[i+1,j-1] -
                       eigval_max[i-1,j+1] + eigval_max[i-1,j-1])/(4*dx*dy)
                
                eigvec_max0 = eigvec_max[i,j,:]
                ex,ey = eigvec_max0
                
                # compute second directional derivative in eigvec direction
                c2 = ex*(fxx*ex + fxy*ey) + ey*(fxy*ex + fyy*ey)
                
                # if second directional derivative is negative and large enough,
                # use taylor expansion of eigval_max to find subpixel ridge point
                if c2 < -sdd_thresh:
                    t = -(fx*ex + fy*ey)/c2
                    # if point lies within current grid, keep point and data
                    if abs(t*ex) <= dx/2 and abs(t*ey) <= dy/2:
                        k = i*ny+j
                        r_pts[k,:2] = pt + np.array([t*ex,t*ey])
                        r_vec[k,:] = eigvec_max0
                        sdd[k] = c2
                        
    return r_pts, r_vec, sdd, h


@njit
def _link_points_stepper(pt0,nvec0,ind0,
                         r_pts,r_vecs,sdd,
                         offset_ravel_inds,
                         ridge_num,a,c=1.,sdd_thresh=0.):
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
    angle_ind = floor((atan2(nvec0[0],-nvec0[1])%(2*pi))*4/pi + 0.5)%8
    offset_inds = offset_ravel_inds[angle_ind:angle_ind+3]
    cand_pts = np.zeros((3,2),numba.float64)
    metric = np.zeros((3),numba.float64)
    pt_vecs = np.zeros((3,2),numba.float64)
    
    # loop through candidate neighbors
    nhbrs_flag = False
    k_off = np.zeros((3,),numba.int32)

    for i in range(3):
        k_off_ = ind0+offset_inds[i]
        k_off[i]= k_off_
        cr_pts = r_pts[k_off_,:]
        cand_pts[i,:] = cr_pts[:2]
        # if neighbor is ridge point, compute linking metric
        if sdd[k_off_] + sdd_thresh < 0:
            d = ((cand_pts[i,0]-pt0[0])**2 + (cand_pts[i,1]-pt0[1])**2)**0.5
            nvec1 = np.array([r_vecs[k_off_,0],
                              r_vecs[k_off_,1]])
            
            dot01 = nvec0[0]*nvec1[0] + nvec0[1]*nvec1[1]

            if dot01 < 0:
                pt_vecs[i,:] = -nvec1
                dot01 = -dot01
            else:
                pt_vecs[i,:] = nvec1
                
            beta = acos(dot01)
            metric[i] = a*d + c*beta
            nhbrs_flag = True
        else:
            metric[i] = 10000.
    
    if nhbrs_flag:
        nhbr_ind = np.argmin(metric)
        pt_ind = k_off[nhbr_ind]
        if r_pts[pt_ind,2] + 0.5 < 0:
            r_pts[pt_ind,2] = ridge_num
            return r_pts,cand_pts[nhbr_ind,:],pt_vecs[nhbr_ind,:],pt_ind
        
        else:
            return r_pts,None,None,None           

    else:
        return r_pts,None,None,None


@njit            
def _linked_ridge_pts(f,eigvec_max,x,y,sdd_thresh=0.,percentile=0,c=1.):
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
    rpts, rvecs, sdd, h = _ftle_ridge_pts_connect(f,eigvec_max,x,y,
                                                  sdd_thresh=sdd_thresh,percentile=percentile)
    
    a = 1/h
    # get indices corresponding to pts sorted by 2nd directional derivative value
    sorted_inds = np.argsort(sdd)
    
    # create raveled version of grid offset inds
    nx,ny = f.shape
    grid_shape = np.array([nx,ny],np.int32)
    offset_inds_arr = np.array([[1,-1],[1,0],[1,1],[0,1],[-1,1],
                                [-1,0],[-1,-1],[0,-1],[1,-1],[1,0]])
    offset_ravel_inds = np.zeros(10,np.int32)
    for i,inds in enumerate(offset_inds_arr):
        offset_ravel_inds[i] = ravel_index(inds,grid_shape)
         
    up_thresh = sdd_thresh + 5
    ridge_num_counter = 0   # counts which ridge
    linked_ridges_arr = np.zeros((nx*ny,2),np.float64)
    ridge_len = np.zeros((nx*ny,2),np.int32)
    endpoints = np.zeros((nx*ny,3),np.float64)
    ep_tanvecs = np.zeros((nx*ny,2),np.float64)
    ridge_pt_counter = 0    # counts how many points
    ridge0_start = 0        # keeps track of index current ridge started at
    # loop through points is descending value (in magnitude) of 2nd directional derivative
    for si in sorted_inds:
        ridge_num = rpts[si,-1]
        # check if current points alreadly belongs to a ridge, if yes, skip
        if ridge_num + 0.5 < 0:
            # check if 2nd directional derivative is above starting threshold
            if sdd[si] < -up_thresh:
                rpts[si,-1] = ridge_num_counter     # set points ridge number to current
                pt0 = rpts[si,:2].copy()            # set starting point, vec, ind
                nvec0 = rvecs[si,:].copy()
                ind0 = si
                
                # set first point in ridge array
                linked_ridges_arr[ridge_pt_counter,:] = np.array([pt0[0],
                                                                  pt0[1]],
                                                                 np.float64)
                
                ridge_pt_counter+=1
                # do I need .copy() here and below?
                
                # set endpoint
                ept0 = pt0.copy()
                
                # find neighbors
                for k in range(10000):
                    rpts,pt0,nvec0,ind0 = _link_points_stepper(pt0, nvec0, ind0,
                                                               rpts, rvecs, sdd, 
                                                               offset_ravel_inds, 
                                                               ridge_num_counter,a,c=c)
                    # if no more neighbors, break
                    if ind0 is None:
                        break
                    
                    # set endpoint
                    ept0 = pt0.copy()
                    
                    # add point to array
                    linked_ridges_arr[ridge_pt_counter,:] = np.array([pt0[0],
                                                                      pt0[1]],
                                                                     np.float64)
                    ridge_pt_counter+=1

                # flip ridge so ridge is in order            
                linked_ridges_arr[ridge0_start:ridge_pt_counter,:] = np.flipud(
                                                                    linked_ridges_arr[ridge0_start:
                                                                                      ridge_pt_counter,:])
                # same process as above but in the other direction
                pt1 = rpts[si,:2].copy()
                ind1 = si
                nvec1 = -rvecs[si,:].copy()
                ept1 = pt1.copy()
                for kk in range(10000):
                    rpts,pt1,nvec1,ind1 = _link_points_stepper(pt1, nvec1, ind1,
                                                               rpts, rvecs, sdd, 
                                                               offset_ravel_inds, 
                                                               ridge_num_counter,a,c=c)
                    
                    if ind1 is None:
                        break
                    ept1 = pt1.copy()
                    linked_ridges_arr[ridge_pt_counter,:] = np.array([pt1[0],
                                                                      pt1[1]],
                                                                     np.float64)
                    ridge_pt_counter+=1
                
                # if ridge only consists of one point, remove from array and reset counter
                # to value on previous iterate
                if ridge0_start == ridge_pt_counter-1:
                    linked_ridges_arr[ridge0_start,:] = np.array([0.,0.])
                    ridge_pt_counter-=1
                    ridge0_start = ridge_pt_counter
                # else, save endpoints, endpoint vecs, and update start ind and ridge num
                else:
                    epvec0 = linked_ridges_arr[ridge0_start+1,:] - ept0
                    epvec0 = epvec0/(epvec0[0]**2 + epvec0[1]**2)**0.5
                    epvec1 = ept1 - linked_ridges_arr[ridge_pt_counter-2,:]
                    epvec1 = epvec1/(epvec1[0]**2 + epvec1[1]**2)**0.5
                    ridge_len[ridge_num_counter,:] = np.array([ridge_pt_counter,
                                                               ridge_pt_counter-ridge0_start])

                    endpoints[2*ridge_num_counter:2*ridge_num_counter+2,:] = np.array([[ept0[0],ept0[1],ridge_num_counter],
                                                                                         [ept1[0],ept1[1],-(ridge_num_counter+0.1)]])
                    ep_tanvecs[2*ridge_num_counter:2*ridge_num_counter+2,:] = np.array([[epvec0[0],epvec0[1]],
                                                                                          [epvec1[0],epvec1[1]]])
                    ridge0_start = ridge_pt_counter
                    ridge_num_counter+=1
                    
                
            else:
                break
            
        else:
            continue
                
    return (linked_ridges_arr[:ridge_pt_counter,:], ridge_len[:ridge_num_counter,:],
            endpoints[:2*ridge_num_counter,:], ep_tanvecs[:2*ridge_num_counter,:])


@njit
def endpoint_distances(pt,arr,dist_tol):
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
    dist = np.zeros(len_arr,numba.float64)
    tan_vec = np.zeros_like(arr,numba.float64)
    tol_bool = np.zeros(len_arr,numba.bool_)
    for k in prange(len_arr):
        tan_vec[k,:] = np.array([arr[k,0]-pt[0],arr[k,1]-pt[1]])
        dist[k] = (tan_vec[k,0]**2 + tan_vec[k,1]**2)**0.5
        if dist[k] < dist_tol:
            tol_bool[k] = 1
            tan_vec[k,:] = tan_vec[k,:]/(tan_vec[k,0]**2 + tan_vec[k,1]**2)**0.5
    return dist, tan_vec, tol_bool



@njit
def _connect_endpoints(endpoint,current_tan_vec,rem_endpoints,rem_endpoints_tan,ep_tan_ang,dist_tol):
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
    
    dist_ep, tan_ep, dist_mask =  endpoint_distances(endpoint[:-1],
                                                     rem_endpoints[:,:-1],dist_tol)
    
    keep_mask = np.ones(rem_endpoints.shape[0],np.bool_)
    connect_ind = -1
    new_endpoint = None
    if dist_mask.any():
        ep_sign = int(copysign(1,endpoint[-1]))
        n_in_tol = dist_mask.sum()
        for k in range(n_in_tol):
            ep_ind = np.argmin(dist_ep)
            new_ep_sign = int(copysign(1,rem_endpoints[ep_ind,-1]))
            tan_ep0 = tan_ep[ep_ind,:]
            new_tan_vec = rem_endpoints_tan[ep_ind,:]

            if (acos(abs(tan_ep0[0]*current_tan_vec[0]+
                         tan_ep0[1]*current_tan_vec[1])) < ep_tan_ang and
                acos(new_ep_sign*(new_tan_vec[0]*tan_ep0[0]+
                                  new_tan_vec[1]*tan_ep0[1])) < ep_tan_ang):
                
                new_ridge_ind = rem_endpoints[ep_ind,-1]
                ep_sign = int(copysign(1,new_ridge_ind))
                connect_ind = new_ridge_ind
                new_endpoint = rem_endpoints[ep_ind+ep_sign,:]
                current_tan_vec = rem_endpoints_tan[ep_ind+ep_sign,:]
                keep_mask[ep_ind] = False
                keep_mask[ep_ind+ep_sign] = False
                rem_endpoints = rem_endpoints[keep_mask,:]
                rem_endpoints_tan = rem_endpoints_tan[keep_mask,:]
                break
            else:
                dist_ep[ep_ind] = 10*dist_tol
            
    return rem_endpoints, rem_endpoints_tan, connect_ind, new_endpoint


def ftle_ridge_curves(f,eigvec_max,x,y,dist_tol,ep_tan_tol=pi/4,min_ridge_len=5,
                      sdd_thresh=0.,percentile=0,c=1.):
    """
    Computes ftle ridge points and links points into ridges, after
    this is done, searching for ridges that should be connected if their
    endpoints are within dist_tol and the angle between the tangent of
    the ridge and vector connecting the endpoints is less than ep_tan_tol.

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
    min_ridge_len : int, optional
        minimum points allowed in ridge after endpoints are connected. The default is 5.
    sdd_thresh : float, optional
        threshold for second directional derivative, should be at least 0.
        The default is 0.
    percentile : int, optional
        percentile of ftle used for min allowed value. The default is 0.
    c : float, optional
        scaling for angle portion of metric. The default is 1..

    Returns
    -------
    ridges : list
        list containing connected, ordered ridges.

    """
    
    ridge_pts, ridge_lens, endpoints, ep_tanvecs = _linked_ridge_pts(f,eigvec_max,x,y,
                                                                     sdd_thresh=sdd_thresh,
                                                                     percentile=percentile,c=c)
    
    
    rem_endpoints = endpoints.copy()
    rem_endpoints_tan = ep_tanvecs.copy()
    nridges = int(endpoints.shape[0]/2)
    ridges = []
    
    # while endpoints remain, search for nearby endpoints
    for k in range(nridges):
        # set current endpoints and tan vectors
        current_endpoints = rem_endpoints[:2,:]
        current_tan_vecs = rem_endpoints_tan[:2,:]
        mask = np.ones(rem_endpoints.shape[0],np.bool_)
        
        # remove current from remaining
        mask[:2] = False
        rem_endpoints = rem_endpoints[mask,:]
        rem_endpoints_tan = rem_endpoints_tan[mask,:]
        
        # compute endpoint distances for current endpoints (both sides)
        dist_ep = np.zeros((2,rem_endpoints.shape[0]),np.float64)
        tan_ep = np.zeros((2,rem_endpoints.shape[0],2),np.float64)
        dist_mask = np.zeros((2,rem_endpoints.shape[0]),np.bool_)
        
        # maybe remove tan_ep
        for i in range(2):
            dist_ep[i,:], tan_ep[i,:,:], dist_mask[i,:] = endpoint_distances(current_endpoints[i,:-1],
                                                                   rem_endpoints[:,:-1],dist_tol)
        
        # tells if other endpoints within dist for both current endpoints
        ep_in_tol = dist_mask[0,:].any() + 2*dist_mask[1,:].any()
        
        # if both endpoints have other endpoints in dist
        if ep_in_tol == 3:
            
            connect_inds0 = np.zeros((nridges,),np.float64)
            ii = 0
            # find which has closer endpoint
            min0 = min(dist_ep[0,:])
            min1 = min(dist_ep[1,:])        
            ep_ind = np.argmin(np.array([min0,min1]))
            # add current endpoint to connect_inds0 and increment
            ridge_ep_ind = current_endpoints[ep_ind,-1]
            connect_inds0[ii] = -(ridge_ep_ind + 0.1)
            ii+=1
            new_endpoint = current_endpoints[ep_ind,:]
            current_tan_vec = current_tan_vecs[ep_ind,:]
            # keep searching for endpoints for current ridge until there are none
            for kk in range(nridges):
                rem_endpoints, rem_endpoints_tan, connect_ind, new_endpoint = _connect_endpoints(
                                                                                new_endpoint,
                                                                                rem_endpoints,
                                                                                rem_endpoints_tan,
                                                                                current_tan_vec,
                                                                                ep_tan_tol,
                                                                                dist_tol)
                # break if no more endpoints
                if new_endpoint is None:
                    break
                
                # add index of endpoints to current ridge
                connect_inds0[ii] = connect_ind
                ii+=1
            
        
    
                   
            
            # do same thing for other endpoint
            connect_inds1 = np.zeros((nridges,),np.float64)
            jj = 0
            ep_sign = int(copysign(1,ridge_ep_ind))
            ep_ind += ep_sign
            # add current endpoint to connect_inds and increment
            ridge_ep_ind = current_endpoints[ep_ind,-1]
            connect_inds1[jj] = -(ridge_ep_ind + 0.1)
            jj+=1
            new_endpoint = current_endpoints[ep_ind,:]
            current_tan_vec = current_tan_vecs[ep_ind,:]
            # keep searching for endpoints for current ridge until there are none
            for kk in range(nridges):
                rem_endpoints, rem_endpoints_tan, connect_ind, new_endpoint = _connect_endpoints(
                                                                                new_endpoint,
                                                                                rem_endpoints,
                                                                                rem_endpoints_tan,
                                                                                current_tan_vec,
                                                                                ep_tan_tol,
                                                                                dist_tol)
                # break if no more endpoints
                if new_endpoint is None:
                    break
                
                # add index of endpoints to current ridge
                connect_inds1[jj] = connect_ind
                jj+=1
            
            # if endpoints are found, add to connect_inds array
            if ii > 1 and jj > 1:
                connect_inds = np.hstack((np.flipud(-(connect_inds0[:ii] + 0.1)),
                                                     connect_inds1[1:jj]))                                
            elif ii > 1:
                connect_inds = connect_inds0[:ii]
            else:
                connect_inds = connect_inds1[:jj]  
            inds = (np.abs(connect_inds)).astype(np.int32)
            
            if ii+jj > 2:
                rinds = ridge_lens[inds,0]
                rlens = ridge_lens[inds,1]
                full_len = rlens.sum()
                if full_len < min_ridge_len:
                    continue
                ridge_arr = np.zeros((full_len,2),np.float64)
                if connect_inds[0] + 0.01 < 0:
                    ridge_arr[0:rlens[0],:] = np.flipud(ridge_pts[rinds[0]-rlens[0]:rinds[0],:])
                else:
                    ridge_arr[0:rlens[0],:] = ridge_pts[rinds[0]-rlens[0]:rinds[0],:]
                j = rlens[0]
                for i,ri in enumerate(connect_inds[1:]):
                    if ri + 0.01 < 0:
                        ridge_arr[j:j+rlens[i+1],:] = np.flipud(ridge_pts[rinds[i+1]-rlens[i+1]:
                                                                          rinds[i+1],:])
                            
                    else:
                        ridge_arr[j:j+rlens[i+1],:] = ridge_pts[rinds[i+1]-rlens[i+1]:rinds[i+1],:]
                        
                    j+=rlens[i+1]
                        
                ridges.append(ridge_arr)
            else:
                rind = ridge_lens[inds,0][0]
                rlen = ridge_lens[inds,1][0]
                if rlen < min_ridge_len:
                    continue
                ridges.append(ridge_pts[rind-rlen:rind,:])
        
        # if only the ending endpoint
        elif ep_in_tol == 2:
            ii = 0
            ep_ind = 1
            ridge_ep_ind = current_endpoints[ep_ind,-1]
            connect_inds = np.zeros((nridges,),np.float64)
            connect_inds[ii] = -(ridge_ep_ind + 0.1)
            ii+=1
            new_endpoint = current_endpoints[ep_ind,:]
            current_tan_vec = current_tan_vecs[ep_ind,:]
            # keep searching for endpoints for current ridge until there are none
            for kk in range(nridges):
                rem_endpoints, rem_endpoints_tan, connect_ind, new_endpoint = _connect_endpoints(
                                                                                new_endpoint,
                                                                                rem_endpoints,
                                                                                rem_endpoints_tan,
                                                                                current_tan_vec,
                                                                                ep_tan_tol,
                                                                                dist_tol)
                # break if no more endpoints
                if new_endpoint is None:
                    break
                
                # add index of endpoints to current ridge
                connect_inds[ii] = connect_ind
                ii+=1
            connect_inds = connect_inds[:ii]
            inds = (np.abs(connect_inds)).astype(np.int32)
            if ii > 1:
                rinds = ridge_lens[inds,0]
                rlens = ridge_lens[inds,1]
                full_len = rlens.sum()
                if full_len < min_ridge_len:
                    continue
                ridge_arr = np.zeros((full_len,2),np.float64)
                if connect_inds[0] + 0.01 < 0:
                    ridge_arr[0:rlens[0],:] = np.flipud(ridge_pts[rinds[0]-rlens[0]:rinds[0],:])
                else:
                    ridge_arr[0:rlens[0],:] = ridge_pts[rinds[0]-rlens[0]:rinds[0],:]
                j = rlens[0]
                for i,ri in enumerate(connect_inds[1:]):
                    if ri + 0.01 < 0:
                        ridge_arr[j:j+rlens[i+1],:] = np.flipud(ridge_pts[rinds[i+1]-rlens[i+1]:
                                                                          rinds[i+1],:])
                            
                    else:
                        ridge_arr[j:j+rlens[i+1],:] = ridge_pts[rinds[i+1]-rlens[i+1]:rinds[i+1],:]
                        
                    j+=rlens[i+1]
                        
                ridges.append(ridge_arr)
            else:
                rind = ridge_lens[inds,0][0]
                rlen = ridge_lens[inds,1][0]
                if rlen < min_ridge_len:
                    continue
                ridges.append(ridge_pts[rind-rlen:rind,:])
        
        # if only the starting endpoint
        elif ep_in_tol == 1:
            ii = 0
            ep_ind = 0
            ridge_ep_ind = current_endpoints[ep_ind,-1]
            connect_inds = np.zeros((nridges,),np.float64)
            connect_inds[ii] = -(ridge_ep_ind + 0.1)
            ii+=1
            new_endpoint = current_endpoints[ep_ind,:]
            current_tan_vec = current_tan_vecs[ep_ind,:]
            # keep searching for endpoints for current ridge until there are none
            for kk in range(nridges):
                rem_endpoints, rem_endpoints_tan, connect_ind, new_endpoint = _connect_endpoints(
                                                                                new_endpoint,
                                                                                rem_endpoints,
                                                                                rem_endpoints_tan,
                                                                                current_tan_vec,
                                                                                ep_tan_tol,
                                                                                dist_tol)
                # break if no more endpoints
                if new_endpoint is None:
                    break
                
                # add index of endpoints to current ridge
                connect_inds[ii] = connect_ind
                ii+=1
            connect_inds = connect_inds[:ii]
            inds = (np.abs(connect_inds)).astype(np.int32)
            if ii > 1:
                rinds = ridge_lens[inds,0]
                rlens = ridge_lens[inds,1]
                full_len = rlens.sum()
                if full_len < min_ridge_len:
                    continue
                ridge_arr = np.zeros((full_len,2),np.float64)
                if connect_inds[0] + 0.01 < 0:
                    ridge_arr[0:rlens[0],:] = np.flipud(ridge_pts[rinds[0]-rlens[0]:rinds[0],:])
                else:
                    ridge_arr[0:rlens[0],:] = ridge_pts[rinds[0]-rlens[0]:rinds[0],:]
                j = rlens[0]
                for i,ri in enumerate(connect_inds[1:]):
                    if ri + 0.01 < 0:
                        ridge_arr[j:j+rlens[i+1],:] = np.flipud(ridge_pts[rinds[i+1]-rlens[i+1]:
                                                                          rinds[i+1],:])
                            
                    else:
                        ridge_arr[j:j+rlens[i+1],:] = ridge_pts[rinds[i+1]-rlens[i+1]:rinds[i+1],:]
                        
                    j+=rlens[i+1]
                        
                ridges.append(ridge_arr)
            else:
                rind = ridge_lens[inds,0][0]
                rlen = ridge_lens[inds,1][0]
                ridges.append(ridge_pts[rind-rlen:rind,:])
        else:
            inds = int(current_endpoints[0,-1])
            rind = ridge_lens[inds,0]
            rlen = ridge_lens[inds,1]
            if rlen < min_ridge_len:
                continue
            ridges.append(ridge_pts[rind-rlen:rind,:])
            
        if rem_endpoints.size <= 4:
            break
        
    return ridges
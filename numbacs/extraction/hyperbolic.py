import numpy as np
from numba import njit, prange
import numba                                 
from ..utils import composite_simpsons_38_irregular, dist_2d, max_in_radius
from interpolation.splines import UCGrid, eval_spline, prefilter
from math import copysign

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

    domain = np.array([[xvals[0],xvals[-1]],[yvals[0],yvals[-1]]])    # for _in_domain
    for direction in np.array([1,-1],np.int32):
        init_direction = direction                  # initialize integration direction
        L=0                                         # initialize failure length
        y = np.zeros((steps+1,2),np.float64)
        y[0,:] = np.array([xvals[ic_ind[0]],yvals[ic_ind[1]]],np.float64) # set initial condition
        init_vec = eigvec_min[ic_ind[0],ic_ind[1],:]    # set initial vector
        a1 = _alpha(y[0,:],eigval_max)              # initial alpha
        i = 0
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
                    init_direction = copysign(1,np.dot(k4,init_vec))
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
                init_direction = copysign(1,np.dot(k4,init_vec))
                
                a1 = _alpha(y[i+1,:],eigval_max)
                if _in_region(y[i+1,:],xvals,yvals,U0):
                    L = 0.
                    remove_inds = 0
                else:
                    L = L + dist_2d(y[i+1,:],y[i,:])
                    remove_inds += 1
                i+=1
        
        i -= remove_inds
        if direction == 1:            
            d1lcs = np.flipud(y[:i+1,:])    # direction 1 candidite lcs
        else:
            d2lcs = y[1:i+1,:]              # direction -1 candidate lcs
            
    tensorline = np.concatenate((d1lcs,d2lcs),axis=0) # concatenate both canditate lcs   
    return tensorline



@njit
def rk4_tensorlines_oecs(eigval_max,eigvec_min,xvals,yvals,ic_ind,h,steps,maxlen,minval):
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
    maxlen : int
         maximum length allowed for OECS curves

    Returns
    -------
    tensorline : np.ndarray, shape = (len(d1lcs) + len(d2lcs),2)
        array containing tensorline representing candidate OECS.

    """

    domain = np.array([[xvals[0],xvals[-1]],[yvals[0],yvals[-1]]])    # for _in_domain
    for direction in np.array([1,-1],np.int32):
        init_direction = direction                  # initialize integration direction
        L=0                                         # initialize failure length
        y = np.zeros((steps+1,2),np.float64)
        y[0,:] = np.array([xvals[ic_ind[0]],yvals[ic_ind[1]]],np.float64) # set initial condition
        init_vec = eigvec_min[ic_ind[0],ic_ind[1],:]    # set initial vector
        s0 = eigval_max(y[0,:])                             # initial eigval
        i = 0
        remove_inds = 0
        monotone_flag = False
        while L<maxlen and i<steps:
            # rk4 to integrate tensorlines from eigvec_min field
            k1 = init_vec
            if init_direction == -1:
                k1 = -k1
            yk2 = y[i,:]+0.5*h*k1
            k2 = _reorient_eigvec(yk2,xvals,yvals,eigvec_min)   # reorient and scale eigenvector
            if np.dot(k1,k2) < 0:                           # make sure direction is continuous
                k2 = -k2
            yk3 = y[i,:]+0.5*h*k2
            k3 = _reorient_eigvec(yk3,xvals,yvals,eigvec_min)
            if np.dot(k2,k3) < 0:
                k3 = -k3
            yk4 = y[i,:]+h*k3
            k4 = _reorient_eigvec(yk4,xvals,yvals,eigvec_min)
            if np.dot(k3,k4) < 0:
                k4 = -k4
            y[i+1,:] = y[i,:] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)   # rk4 step
            L+=h
            s1 = eigval_max(y[i+1,:])
            # check if point is in domain and not stuck on boundary or fixed point
            if s1 < minval:
                break
            if _in_domain(y[i+1,:],domain) == False:
                break
            if s1 > s0:
                remove_inds+=1
                monotone_flag = True
                break
            s0 = s1
            if i > 1:
                r0 = y[i,:] - y[i-1,:]
                r1 = y[i+1,:] - y[i,:]
                if np.dot(r0,r1) >= 0:
                    # set updated init_vector and init_direction
                    init_vec = _reorient_eigvec(y[i+1,:],xvals,yvals,eigvec_min)
                    init_direction = np.sign(np.dot(k4,init_vec))
                    i+=1
                else:
                    break
            else:
                # set updated init_vector and init_direction
                init_vec = _reorient_eigvec(y[i+1,:],xvals,yvals,eigvec_min)
                init_direction = np.sign(np.dot(k4,init_vec))
                i+=1
        
        i -= remove_inds
        if i < 10 and monotone_flag:
            return None
        
        if direction == 1:            
            d1oecs = np.flipud(y[:i+1,:])    # direction 1 candidite oecs
        else:
            d2oecs = y[1:i+1,:]              # direction -1 candidate oecs
            
    tensorline = np.concatenate((d1oecs,d2oecs),axis=0) # concatenate both canditate oecs   
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
    npts = len(pts)
    arclength = np.zeros(npts-1,numba.float64)
    for k in prange(npts-1):
        p0 = pts[k,:]
        p1 = pts[k+1,:]
        arclength[k] = ((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)**0.5
        
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
def _hyp_lcs(tensorlines,lambda_avg,vlines,hlines,dist_tol=1e-1):
    """
    Compute hyperbolic LCS tensorlines from 'rk4_tensorlines', compare nearby tensorlines
    to see which candidatelcs curves are the most attracting/repelling.

    Parameters
    ----------
    tensorlines : np.ndarray, shape = (max_tensorline_len+1,2*nt)
        array containing tensorlines.
    lambda_avg : np.ndarray, shape = (nt,)
        array containing the average attraction/repulsion rate for each tensorline.
    vlines : np.ndarray, shape = (nlinesx,)
        array containing values of vertical lines.
    hlines : np.ndarray, shape = (nlinesy,)
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
    keep_inds = -1*np.ones(nt,np.int32)    # prealocate array for inds to keep
    # keep_inds_bool = np.zeros(nt,numba.bool_)
    # keep_inds = np.arange(0,nt,numba.int32)
    jj = 0
    for vl in vlines:
        v_intersect = np.zeros((nt*max_intersects,3),np.float64) # prealocate 
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
        v_intersect = v_intersect[:ii,:]
        for k in range(ii):
            t_ind = int(v_intersect[k,0])   # tensorline index corresponding to intersection             
            ot_inds = v_intersect[:,0] != t_ind     # find all other tensorlines intersecting vl
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
        h_intersect = h_intersect[:ii,:]
        for k in range(ii):
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


@njit(parallel=parallel_flag)
def _lcs_region(eigval_max,eigvec_max,dx,dy,percentile=0):
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
    percentile : int, optional
        percentile of eigval_max used for min allowed value. The default is 0.        

    Returns
    -------
    lcs_reg : np.ndarray, shape = (nx,ny)
        0,1 array which states if grid point satisfies LCS condition.

    """
    if percentile == 0:
        eigval_min_val = 1
    else:
        eigval_min_val = np.percentile(eigval_max,percentile)
    nx,ny = eigval_max.shape
    lcs_reg = np.zeros((nx,ny),numba.int32)
    for i in prange(2,nx-2):
        for j in range(2,ny-2):
            if eigval_max[i,j] <= eigval_min_val:
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


@njit
def _endpoint_distances_lcs(pt,arr,dist_tol):
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
    tol_bool : np.ndarray, shape = (npts,)
        truth values which deterimines if kth point is within dist_tol of pt.
        
    """
    
    len_arr = arr.shape[0]
    dist = np.zeros(len_arr,numba.float64)
    tol_bool = np.zeros(len_arr,numba.bool_)
    for k in prange(len_arr):
        dist[k] = ((arr[k,0]-pt[0])**2 + (arr[k,1]-pt[1])**2)**0.5
        if dist[k] < dist_tol:
            tol_bool[k] = 1
    return dist, tol_bool


def _compute_lcs(eigval_max,eigvecs,x,y,h,steps,lf,lmin,r,nmax,lambda_avg_min,percentile=0):
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
    lambda_avg_min : int
        minimum allowed value for lambda_avg for curve to be considered for lcs.
    percentile : int, optional
        percentile of eigval_max used for min allowed value. The default is 0.

    Returns
    -------
    lcs_arr : np.ndarray, shape = (2*steps,2*k)
        array containing candidate LCS tensorlines.
    lambda_avg : np.ndarray, shape = (k,)
        average attraction/repullsion rate of each candidate LCS.
    arclength : np.ndarray, shape = (k,)
        arclength of each candidate LCS.
    endpoints : np.ndarray, shape = (2*k,3)
        array containing endpoints of each curve and curve number.

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
    
    U0 = _lcs_region(eigval_max,eigvecs[:,:,:,1],dx,dy,percentile=percentile)
    nic = len(max_inds)
    lcs_arr = np.zeros((2*steps+2,2*nic),np.float64)
    lambda_avg = np.zeros(nic,np.float64)
    arclength = np.zeros(nic,np.float64)
    endpoints = np.zeros((2*nic,3),np.float64)
    k = 0
    for ic_ind in max_inds:
        clcs = rk4_tensorlines(eigval_max_spline,eigvecs[:,:,:,0],x,y,ic_ind,h,steps,U0,lf)
        arclength_arr = _arclength_arr(clcs)
        arclength_tmp = arclength_arr.sum()
        clcs_lambda = eigval_max_spline(clcs)
        if arclength_tmp >= lmin:
            lambda_avg_tmp = composite_simpsons_38_irregular(clcs_lambda,arclength_arr)/arclength_tmp
            if lambda_avg_tmp >= lambda_avg_min:
                lambda_avg[k] = lambda_avg_tmp
                endpoints[2*k,:] = np.array([clcs[0,0],clcs[0,1],k])
                endpoints[2*k+1,:] = np.array([clcs[-1,0],clcs[-1,1],k])
                llen = len(clcs)
                lcs_arr[:llen,2*k:2*k+2] = clcs
                lcs_arr[-1,2*k] = llen
                arclength[k] = arclength_tmp
                k+=1
        del clcs
    lambda_avg = lambda_avg[:k]
    lcs_arr = lcs_arr[:,:2*k]
    arclength = arclength[:k]
    endpoints = endpoints[:2*k,:]
        
    return lcs_arr, lambda_avg, arclength, endpoints
        
def hyperbolic_lcs(eigval_max,eigvecs,x,y,h,steps,lf,lmin,r,nmax,dist_tol,nlines,
                lambda_avg_min=0,percentile=0,ep_dist_tol=0.0,arclen_flag=False):
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
    dist_tol : float
        distance tolerance used when comparing candidate LCS.
    nlines : int
        number of lines in dimension with largest length used when comparing
        candidate LCS, number of lines in other dimension will be scaled by
        the ratio of the dimensions lengths.    
    lambda_avg_min : int
        minimum allowed value for lambda_avg for curve to be considered for lcs. The default is 0.
    percentile : int, optional
        percentile of eigval_max used for min allowed value. The default is 0.  
    ep_dist_tol : float, optional
        value used to compare starting/ending points of curves, if endpoints are within
        this value, all curves with smaller lambda_avg values are discarded. If
        ep_dist_tol < 0.0, this step is skipped and endpoints are not compared. The
        default is 0.0.
    arclen_flag : bool, optional
        flag to determine if the average attraction/repulsion rate along curve is used
        or if the total attraction/repulsion rate for a curve is used for comparison.
        The default is False.

    Returns
    -------
    lcs : list
        collection of most attracting/repelling LCS.

    """
    lcs = []
    lcs_arr_, lambda_avg_, arclength_, endpoints_ = _compute_lcs(eigval_max,eigvecs,x,y,h,steps,
                                                                 lf,lmin,r,nmax,lambda_avg_min,
                                                                 percentile=percentile)    
    
    if arclen_flag:
        lambda_avg_ = lambda_avg_*arclength_
    if ep_dist_tol > 0: 
        nt = len(lambda_avg_)
        prefilter_bool = np.zeros(nt,np.bool_)
        prefilter_inds  = np.arange(0,2*nt).reshape(nt,2)
        while endpoints_.size > 0:
            ep = endpoints_[0,:-1]
            ep_ind = round(endpoints_[0,2])
            ep_dist, d_bool = _endpoint_distances_lcs(ep,endpoints_,ep_dist_tol)
            near_inds = np.round(endpoints_[d_bool,2]).astype(np.int32)
            max_ind = np.argmax(lambda_avg_[near_inds])
            prefilter_bool[near_inds[max_ind]] = True
            if len(near_inds) == 1:
                endpoints_ = endpoints_[1:,:]
                continue
            del_inds = np.isin(endpoints_[:,2],np.delete(near_inds,max_ind))
            if max_ind == ep_ind:
                endpoints_ = endpoints_[~del_inds,:][1:,:]
            else:
                endpoints_ = endpoints_[~del_inds,:]
            
        new_inds_2d = prefilter_inds[prefilter_bool,:].ravel()
        lcs_arr_ = lcs_arr_[:,new_inds_2d]
        lambda_avg_ = lambda_avg_[prefilter_bool]
    if dist_tol > 0:
        xlen = x[-1] - x[0]
        ylen = y[-1] - y[0]
        if xlen - ylen >= 0:
            nlinesx = nlines
            nlinesy = int(nlines*ylen/xlen)
        else:
            nlinesx = int(nlines*xlen/ylen)
            nlinesy = nlines
        
        vlines = np.linspace(x[2],x[-3],nlinesx)
        hlines = np.linspace(y[2],y[-3],nlinesy)
        lcs_keep_inds = _hyp_lcs(lcs_arr_,lambda_avg_,vlines,hlines,dist_tol)
    else:
        lcs_keep_inds = np.arange(0,len(lambda_avg_))
    for ind in lcs_keep_inds:
        lcs_ilen = int(lcs_arr_[-1,2*ind])
        lcs.append(lcs_arr_[:lcs_ilen,2*ind:2*ind+2])
        
    return lcs


@njit
def _hyperbolic_oecs(eigval_max_interp,eigvecs,x,y,ic_inds,h,steps,maxlen,minval):
    """
    

    Parameters
    ----------
    eigval_max_interp : jit-callable
        jit-callable interpolant of eigval_max of S, linear interpolant is recommended.
    eigvecs : np.ndarray, shape = (nx,ny,2,2)
        array containing both minimum and maximum eigenvectors of S.
    x : np.ndarray, shape = (nx,)
        array containing x-values.
    y : np.ndarray, shape = (ny,)
        array containing x-values.
    ic_inds : np.ndarray, shape = (ni,2)
        array of initial conditions.
    h : float
        step size used in Runge-Kutta solver.
    steps : int
        maximum number of steps for Runge-Kutta solver.
    maxlen : float
        maximum allowed length for oecs curve.
    minval : float
        minimum value allowed for eigval_max.

    Returns
    -------
    oecs_fwd : np.ndarray, shape = (len(keep_inds),2*steps+1,2)
        array containing forward oecs.
    oecs_bwd : np.ndarray, shape = (len(keep_inds),2*steps+1,2)
        array containing backward oecs.
    centers : np.ndarray, shape = (len(keep_inds),2)
        array containing the center or saddle point of oecs.

    """
    
    eigvec_min = eigvecs[:,:,:,0]
    eigvec_max = eigvecs[:,:,:,1]
    ni = ic_inds.shape[0]
    oecs_fwd = np.zeros((ni,2*steps+1,2),np.float64)
    oecs_bwd = np.zeros((ni,2*steps+1,2),np.float64)
    centers = np.zeros((ni,2),np.float64)
    keep_inds = np.zeros(ni,np.bool_)
    for k in prange(ni):
        ic = ic_inds[k,:]
        oecs_fwd_tmp = rk4_tensorlines_oecs(eigval_max_interp,eigvec_min,x,y,ic,h,steps,
                                            maxlen,minval)
        if oecs_fwd_tmp is not None:
            tmplen = oecs_fwd_tmp.shape[0]
            oecs_fwd[k,:tmplen,:] = oecs_fwd_tmp
            oecs_fwd[k,-1,0] = tmplen
            
            oecs_bwd_tmp = rk4_tensorlines_oecs(eigval_max_interp,eigvec_max,x,y,ic,h,steps,
                                                maxlen,minval)
            if oecs_bwd_tmp is not None:
                keep_inds[k] = True
                tmplen = oecs_bwd_tmp.shape[0]
                oecs_bwd[k,:tmplen,:] = oecs_bwd_tmp
                oecs_bwd[k,-1,0] = tmplen
                centers[k,:] = np.array([x[ic[0]],y[ic[1]]])
            
    return oecs_fwd[keep_inds,:,:], oecs_bwd[keep_inds,:,:], centers[keep_inds,:]
    

def hyperbolic_oecs(eigval_max,eigvecs,x,y,r,h,steps,maxlen,minval,n=-1):
    """
    Wrapper for _hyperbolic_oecs that returns list of oecs saddles.

    Parameters
    ----------
    eigval_max : np.ndarray, shape = (nx,ny)
        array containing values of max eigenvalue of S.
    eigvecs : np.ndarray, shape = (nx,ny,2,2)
        array containing both minimum and maximum eigenvectors of S.
    x : np.ndarray, shape = (nx,)
        array containing x-values.
    y : np.ndarray, shape = (ny,)
        array containing x-values.
    r : float
        radius used to find local maxima.
    h : float
        step size used in Runge-Kutta solver.
    steps : int
        maximum number of steps for Runge-Kutta solver.
    maxlen : float
        maximum allowed length for oecs curve.
    minval : float
        minimum value allowed for eigval_max.
    n : int, optional
        number of local maxima to look for, if n = -1, all are used. The default is -1.

    Returns
    -------
    oecs_saddles : list
        list containing oecs saddles defined by oecs forward and oecs backward curves.

    """
    
    oecs_saddles = []
    
    nx,ny = eigval_max.shape
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    _, ic_inds = max_in_radius(eigval_max.copy(),r,dx,dy,n=n)
    
    grid = UCGrid((x[0],x[-1],nx),(y[0],y[-1],ny))
    @njit
    def eigval_max_interp(point):
        return eval_spline(grid,eigval_max,point,out=None,k=1,diff="None",extrap_mode="constant")
    
    oecs_fwd, oecs_bwd, centers = _hyperbolic_oecs(eigval_max_interp,eigvecs,x,y,
                                          ic_inds,h,steps,maxlen,minval)
    
    for k in range(oecs_fwd.shape[0]):
        oecs_len = int(oecs_fwd[k,-1,0])
        oecs_fwd_tmp = oecs_fwd[k,:oecs_len,:]
        oecs_len = int(oecs_bwd[k,-1,0])
        oecs_bwd_tmp = oecs_bwd[k,:oecs_len,:]
        oecs_saddles.append([oecs_fwd_tmp,oecs_bwd_tmp,centers[k,:]])
        
    return oecs_saddles

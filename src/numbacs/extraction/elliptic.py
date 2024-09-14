import numpy as np
from numba import njit                                 
from ..utils import max_in_radius, shoelace, pts_in_poly, arclength
from contourpy import contour_generator
from scipy.spatial import ConvexHull



def rotcohvrt(lavd,x,y,r,convexity_method='convex_hull',convexity_deficiency=5e-3,min_val=-1.0,
        nlevs=20,start_level=0.0,end_level=0.0,min_len=0.0):
    """
    Compute rotationally coherent vortices which are (approximately) convex closed
    contours of the lavd (or ivd) field.

    Parameters
    ----------
    lavd : np.ndarray, shape = (nx,ny)
        array containing lavd values.
    x : np.ndarray, shape = (nx,)
        array containing x-values.
    y : np.ndarray, shape = (ny,)
        array containing y-values.
    r : float
        radius in which points will be discared after a maximum is found at the center.
    convexity_method : str, optional
        method used to determine if closed curve is convex, options are
        'convex_hull' which calculates the relative difference in area of the curve and its
        convex hull, 'angle' which checks if angle between succesive edges exceeds
        convexity_deficiency in the negative direction. The default is 'convex_hull'.
    convexity_deficiency : float, optional
        value used to allow convexity deficiency up to this value, if 'convex_hull' method
        is used this is the allowable relative difference in area, if 'angle' method is used
        this is the allowable angle. The default is 5e-3.
    min_val : float, optional
        minimum value allowed for maxima of the lavd field, if equals -1.0, the 80th percentile
        of the lavd is used. The default is -1.0.
    nlevs : int, optional
        number of levels of contours to use when searching for convex curves. The default is 20.
    start_level : float, optional
        starting level of contours for search, if equals 0.0 then the 70th percentile
        of the lavd is used. The default is 0.0.
    end_level : float, optional
        value for last contour level to be checked, if equals 0.0 then the maximum of the lavd
        is used. The default is 0.0.

    Returns
    -------
    rcv : list
        list containing rotationally coherent vortices and corresponding vortex centers.

    """
    
    
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    if min_val == -1.0:
        min_val = np.percentile(lavd,80)
    
    max_vals, max_inds = max_in_radius(lavd.copy(),r,dx,dy,min_val = min_val)
        
    if start_level == 0.0:
        start_level = np.percentile(lavd,70)
    
    if end_level == 0.0:
        end_level = max(max_vals)
        
    clevels = np.linspace(start_level,end_level,nlevs)
    
    rem_max_pts = np.column_stack((x[max_inds[:,0]],y[max_inds[:,1]]))
    nrem = len(rem_max_pts)
    rcv = []
    c = contour_generator(x=x,y=y,z=lavd.T)
    if min_len:
        for k in range(nlevs):
            ck = c.lines(clevels[k])
            for contour in ck:
                if contour[0,0] == contour[-1,0] and contour[0,1] == contour[-1,1]:
                    if arclength(contour) > min_len:
                        ind = pts_in_poly(contour,rem_max_pts)
                        if ind >= 0:
                            hull = ConvexHull(contour)
                            ch = contour[np.hstack((hull.vertices,hull.vertices[0])),:]
                            area = shoelace(contour)
                            if (hull.volume - area)/area < convexity_deficiency:
                                rcv.append([ch,rem_max_pts[ind,:]])
                                mask = np.ones(nrem,np.bool_)
                                mask[ind] = False
                                rem_max_pts = rem_max_pts[mask,:]
                                nrem = len(rem_max_pts)
                                if nrem == 0:
                                    
                                    return rcv
    else:
        for k in range(nlevs):
            ck = c.lines(clevels[k])
            for contour in ck:
                if contour[0,0] == contour[-1,0] and contour[0,1] == contour[-1,1]:
                    ind = pts_in_poly(contour,rem_max_pts)
                    if ind >= 0:
                        hull = ConvexHull(contour)
                        ch = contour[np.hstack((hull.vertices,hull.vertices[0])),:]
                        area = shoelace(contour)
                        if (hull.volume - area)/area < convexity_deficiency:
                            rcv.append([ch,rem_max_pts[ind,:]])
                            mask = np.ones(nrem,np.bool_)
                            mask[ind] = False
                            rem_max_pts = rem_max_pts[mask,:]
                            nrem = len(rem_max_pts)
                            if nrem == 0:
                                
                                return rcv                                
                        
                            
                            
    return rcv

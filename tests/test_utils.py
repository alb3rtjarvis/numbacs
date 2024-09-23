import pytest
import numpy as np
from math import pi, isclose
from numbacs.utils import (unravel_index, ravel_index, curl_vel, curl_func_tspan,
                           composite_simpsons, composite_simpsons_38_irregular, dist_2d,
                           dist_tol, shoelace, max_in_radius, gen_circ, gen_filled_circ,
                           gen_filled_circ_radius, arclength, arclength_along_arc,
                           interp_curve, wn_pt_in_poly, pts_in_poly, pts_in_poly_mask,
                           cart_prod)


@pytest.fixture
def polygon():
    
    poly = np.array([[ 0.1757813, -2.2625953],
                    [ 2.2961426, -1.2084065],
                    [ 2.2631836,  0.966751 ],
                    [ 1.0766602,  2.3613918],
                    [-1.1535645,  2.6577378],
                    [-2.9443359,  1.5598659],
                    [-3.5046387, -0.0878906],
                    [-2.9992676, -1.3841427],
                    [-1.6589355, -0.0109863],
                    [-0.736084 ,  0.1977535],
                    [-0.1757813, -0.4394488],
                    [ 0.1757813, -2.2625953]])
    
    return poly

@pytest.fixture
def polygon_tpoints():
    
    pt_in = np.array([-0.4174805, 0.9777357])
    pt_out = np.array([-1.0327148, -0.3955047])
    
    pts = np.column_stack((pt_in,pt_out)).T
    
    return pts


def test_unravel_index():
    
    inds_expected = np.array([3,  8, 17])
    
    index = 777
    shape = np.array([5,10,20])
    inds = unravel_index(index,shape)
    
    assert np.allclose(inds,inds_expected)
    
    
def test_ravel_index():
    
    index_expected = 777
    
    inds = np.array([3,  8, 17])
    shape = np.array([5, 10, 20])
    index = ravel_index(inds,shape)
    
    assert np.allclose(index,index_expected)
    
 
def test_curl_vel(vel_data, vort_data):
    
    dx = 0.1
    u,v = vel_data
    
    vort = curl_vel(u,v,dx,dx)
    
    assert np.allclose(vort, vort_data[0,:,:])
    

def test_curl_func_tspan(coords_dg, vort_data, flow_callable):
    
    dx = 0.1
    x,y = coords_dg
    
    vort = curl_func_tspan(flow_callable, np.array([0.]), x, y, h = dx)
    
    assert np.allclose(vort, vort_data[0,:,:])


def test_composite_simpsons():
    
    expected = -1.3710188464061064
    
    h = 0.04
    x = np.arange(0,1+h,h)
    f = x**3 + 2*x**2 -5*x + np.sin(3*pi*x)
    
    int_val = composite_simpsons(f,h)
    
    assert isclose(int_val,expected)

    
def test_composite_simpsons_38_irregular():
    
    expected = -1.3700178504134466
    
    h = 0.04
    x = np.arange(0,1+h,h)**2
    f = x**3 + 2*x**2 -5*x + np.sin(3*pi*x)
    
    int_val = composite_simpsons_38_irregular(f,np.diff(x))
    
    assert isclose(int_val,expected)
    

def test_dist_2d():
    
    dist_expected = 5.456189146281496
    
    p1 = np.array([0.1,0.5])
    p2 = np.array([-2.3,5.4])
    
    dist = dist_2d(p1,p2)
    
    assert isclose(dist,dist_expected)
    
    
def test_dist_tol():
    
    bool_expected = np.array([True,False])
    
    arr = np.array([[0.44974878, 0.74474145],
                    [0.97850445, 0.01499235]])
    pt = np.array([0.1,0.5])
    tol1 = 0.5
    tol2 = 0.1
    
    bool_arr = np.zeros(2,np.bool_)
    bool_arr[0] = dist_tol(pt,arr,tol1)
    bool_arr[1] = dist_tol(pt,arr,tol2)
    
    assert np.all(bool_arr == bool_expected)
    
    
def test_shoelace(polygon):

    area_expected = 17.199448152554602
    
    area = shoelace(polygon)

    assert isclose(area,area_expected)
    
    
def test_max_in_radius():
    
    maxvals_expected = np.array([0.96984631, 0.96984631])
    maxinds_expected = np.array([[2, 7],
                                 [7, 2]],np.int32)
    
    x = np.linspace(-pi,pi,10)
    y = np.linspace(-pi,pi,10)
    X,Y = np.meshgrid(x,y,indexing='ij')
    f = np.sin(X)*np.cos(Y + pi/2)
    
    r = 2.5
    dx = x[1] - x[0]
    dy = dx
    
    maxvals, maxinds = max_in_radius(f.copy(), r, dx, dy)
    
    assert (np.allclose(maxvals, maxvals_expected)
            & np.all(maxinds == maxinds_expected))
    
    
def test_gen_circ():

    pts_expected = np.array([[ 1.4       ,  0.2       ],
                            [ 1.04906666,  1.16418141],
                            [ 0.16047227,  1.67721163],
                            [-0.85      ,  1.49903811],
                            [-1.50953893,  0.71303021],
                            [-1.50953893, -0.31303021],
                            [-0.85      , -1.09903811],
                            [ 0.16047227, -1.27721163],
                            [ 1.04906666, -0.76418141],
                            [ 1.4       ,  0.2       ]])

    r = 1.5
    c = np.array([-0.1,0.2])
    n = 10
    
    pts = gen_circ(r, c, n)   
    
    assert np.allclose(pts, pts_expected)
    
    
def test_gen_filled_circ():

    pts_expected = np.array([[-0.44976477,  0.52041318],
                            [ 0.03113859, -1.29425656],
                            [ 0.81265829,  1.39040113],
                            [-1.57707023, -0.06127293],
                            [ 1.16563294, -0.60509208],
                            [-0.48940646,  1.64857261],
                            [-0.79136054, -1.13117264],
                            [ 1.30898194,  0.71455795],
                            [-1.48651833,  0.77233461],
                            [ 0.53576899, -1.15860141]])

    r = 1.5
    c = np.array([-0.1,0.2])
    n = 10
    
    pts = gen_filled_circ(r, n, c=c)   
    
    assert np.allclose(pts, pts_expected)
    
    
def test_gen_filled_circ_radius():

    pts_expected = np.array([[-0.44976477,  0.52041318],
                            [ 0.03113859, -1.29425656],
                            [ 0.81265829,  1.39040113],
                            [-1.57707023, -0.06127293],
                            [ 1.16563294, -0.60509208],
                            [-0.48940646,  1.64857261],
                            [-0.79136054, -1.13117264],
                            [ 1.30898194,  0.71455795],
                            [-1.48651833,  0.77233461],
                            [ 0.53576899, -1.15860141]])
    
    radius_expected =np.array([0.47434165, 1.5, 1.5, 1.5,
                               1.5, 1.5,1.5, 1.5, 1.5, 1.5])

    r = 1.5
    c = np.array([-0.1,0.2])
    n = 10
    
    pts, radius = gen_filled_circ_radius(r, n, c=c)   
    
    assert np.allclose(pts, pts_expected) & np.allclose(radius, radius_expected)
    
    
def test_arclength(polygon):

    alen_expected = 19.426768853864864

    alen = arclength(polygon)  
    
    assert isclose(alen, alen_expected)
    
    
def test_arclength_along_arc(polygon):

    alen_expected = np.array([0.,  2.36796243,  4.54336962,  6.37445147,
                              8.62427883, 10.72479899, 12.46521294, 13.85649632,
                              15.77536282, 16.72152728, 17.57003532, 19.42676885])

    alen_arr = arclength_along_arc(polygon)  
    
    assert np.allclose(alen_arr, alen_expected)
    
    
def test_interp_curve(polygon):
    
    ci_expected = np.array([[ 0.1757813 , -2.2625953 ],
                            [ 1.71709707, -1.94150504],
                            [ 2.4221318 , -0.82829153],
                            [ 2.38851236,  0.58940304],
                            [ 1.71831959,  1.83900836],
                            [ 0.55766116,  2.5932545 ],
                            [-0.85058282,  2.71429713],
                            [-2.17857259,  2.22246231],
                            [-3.16435293,  1.26480539],
                            [-3.50256574, -0.11599247],
                            [-2.98871074, -1.38698711],
                            [-2.10143191, -0.46362852],
                            [-0.79787071,  0.21738928],
                            [ 0.00512779, -0.97677892],
                            [ 0.1757813 , -2.2625953 ]])
    
    ci = interp_curve(polygon, 15)
    
    assert np.allclose(ci, ci_expected)
    


def test_wn_pt_in_poly(polygon, polygon_tpoints):
    
    bool_expected = np.array([1,0])
    
    pt_in = polygon_tpoints[0,:]
    pt_out = polygon_tpoints[1,:]
    
    bool_arr = np.zeros(2,int)
    bool_arr[0] = wn_pt_in_poly(polygon,pt_in)
    bool_arr[1] = wn_pt_in_poly(polygon,pt_out)
    
    assert np.all(bool_arr == bool_expected)
    
    
def test_pts_in_poly(polygon, polygon_tpoints):
    
    ind_expected = np.array([0,-1])
    
    pts0 = polygon_tpoints
    pts1 = pts0.copy()
    pts1[0,:] = np.array([-1.9775391, -2.7894248])
    
    ind_arr = np.zeros(2,int)
    ind_arr[0] = pts_in_poly(polygon,pts0)
    ind_arr[1] = pts_in_poly(polygon,pts1)
    
    assert np.all(ind_arr == ind_expected)
  
    
def test_pts_in_poly_mask(polygon, polygon_tpoints):
    
    bool_expected = np.array([True,False])
    
    pts = polygon_tpoints
    
    bool_arr = pts_in_poly_mask(polygon,pts)
    
    assert np.all(bool_arr == bool_expected)
    
    
def test_cart_prod():
    
    cprod_expected = np.array([[0, 0, 0],
                               [0, 0, 1],
                               [0, 0, 2],
                               [0, 1, 0],
                               [0, 1, 1],
                               [0, 1, 2]])
    
    t = np.arange(0,1)
    x = np.arange(0,2)
    y = np.arange(0,3)
    
    cprod = cart_prod((t,x,y))
    
    assert np.all(cprod == cprod_expected)
    

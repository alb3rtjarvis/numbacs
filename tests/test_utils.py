import pytest
import numpy as np
from math import pi, isclose
from numbacs.utils import (
    gradF_stencil_2D,
    gradF_aux_stencil_2D,
    gradF_main_stencil_2D,
    gradUV_stencil_2D,
    eigvalsh_max_2D,
    inv_2D,
    vec_dot_2D,
    vec_dot_3D,
    unravel_index,
    ravel_index,
    curl_vel,
    curl_func_tspan,
    composite_simpsons,
    composite_simpsons_38_irregular,
    dist_2d,
    dist_tol,
    shoelace,
    max_in_radius,
    gen_circ,
    gen_filled_circ,
    gen_filled_circ_radius,
    arclength,
    arclength_along_arc,
    interp_curve,
    wn_pt_in_poly,
    pts_in_poly,
    pts_in_poly_mask,
    cart_prod,
    lonlat2xyz,
    local_basis_S2
)


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

@pytest.fixture
def lonlat_coords():

    lon = np.linspace(-180.0, 180.0, 5)
    lat = np.linspace(-90, 90, 3)

    return np.meshgrid(lon, lat, indexing='ij')

def test_gradF_stencil_2D(coords_dg, fm_data):

    x, y = coords_dg
    dx = x[1]
    dy = y[1]

    gradF_expected = (
        -5.976700186729431,
        -5.0307804346084595,
        0.331292524933815,
        1.0545916110277176
    )
    gradF = gradF_stencil_2D(fm_data, 5, 5, dx, dy)

    assert np.allclose(gradF, gradF_expected)


def test_gradF_aux_stencil_2D(coords_dg, fm_aux_data):

    gradF_aux_expected = (
        -7.270276546478271,
         -9.016692638397217,
         0.5433335900306702,
         0.5362555384635925
    )
    gradF_aux = gradF_aux_stencil_2D(fm_aux_data, 5, 5, 1e-5)

    assert np.allclose(gradF_aux, gradF_aux_expected)


def test_gradF_main_stencil_2D(coords_dg, fm_aux_data):

    x, y = coords_dg
    dx = x[1]
    dy = y[1]

    gradF_main_expected = (
        -5.976700186729431,
        -5.0307804346084595,
        0.331292524933815,
        1.0545916110277176
    )
    gradF_main = gradF_main_stencil_2D(fm_aux_data, 5, 5, dx, dy)

    assert np.allclose(gradF_main, gradF_main_expected)


def test_gradUV_stencil_2D(coords_dg, vel_data):

    x, y = coords_dg
    dx = x[1]
    dy = y[1]

    gradUV_expected = (0.0, 0.9708055193627335, -0.9708055193627335, 0.0)
    u, v = vel_data
    gradUV = gradUV_stencil_2D(u, v, 5, 5, dx, dy)

    assert np.allclose(gradUV, gradUV_expected)

def test_eigvalsh_max_2D():

    B = np.array([[1.3, -2.4], [0.7, 5.2]])
    A = B.T @ B

    eigval_expected = 32.8088282841737
    eigval = eigvalsh_max_2D(A)

    assert np.allclose(eigval, eigval_expected)

def test_inv_2D():

    B = np.array([[1.3, -2.4], [0.7, 5.2]])
    B_singular = B.copy()
    B_singular[:, 0] = 0.0

    Binv_expected = np.array([[0.61611374,  0.28436019], [-0.08293839,  0.15402844]])
    Bsinv_expected = np.array([[0.0, 0.0], [0.0, 0.0]])

    Binv = inv_2D(B)
    Bsinv = inv_2D(B_singular)

    assert np.allclose(Binv, Binv_expected) and np.allclose(Bsinv, Bsinv_expected)

def test_vec_dot_2D():

    v1 = np.array([1.3, -2.4])
    v2 = np.array([0.7, 5.2])

    dot_expected = -11.57
    dot = vec_dot_2D(v1, v2)

    assert np.allclose(dot, dot_expected)


def test_vec_dot_3D():

    v1 = np.array([1.3, -2.4, 3.2])
    v2 = np.array([0.7, 5.2, -2.1])

    dot_expected = -18.29
    dot = vec_dot_3D(v1, v2)

    assert np.allclose(dot, dot_expected)


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


def test_lonlat2xyz(lonlat_coords):

    Lon, Lat = lonlat_coords

    xyz_points_expected = np.array([
           [[-1.22464680e-16, -1.49975978e-32, -2.00000000e+00],
            [-2.00000000e+00, -2.44929360e-16,  0.00000000e+00],
            [-1.22464680e-16, -1.49975978e-32,  2.00000000e+00]],

           [[ 7.49879891e-33, -1.22464680e-16, -2.00000000e+00],
            [ 1.22464680e-16, -2.00000000e+00,  0.00000000e+00],
            [ 7.49879891e-33, -1.22464680e-16,  2.00000000e+00]],

           [[ 1.22464680e-16,  0.00000000e+00, -2.00000000e+00],
            [ 2.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [ 1.22464680e-16,  0.00000000e+00,  2.00000000e+00]],

           [[ 7.49879891e-33,  1.22464680e-16, -2.00000000e+00],
            [ 1.22464680e-16,  2.00000000e+00,  0.00000000e+00],
            [ 7.49879891e-33,  1.22464680e-16,  2.00000000e+00]],

           [[-1.22464680e-16,  1.49975978e-32, -2.00000000e+00],
            [-2.00000000e+00,  2.44929360e-16,  0.00000000e+00],
            [-1.22464680e-16,  1.49975978e-32,  2.00000000e+00]]
       ]
    )

    xyz_points = lonlat2xyz(Lon, Lat, 2.0, deg2rad=True, return_array=True)

    assert np.allclose(xyz_points, xyz_points_expected)


def test_local_basis_S2(lonlat_coords):

    Lon, Lat = lonlat_coords

    basis_expected = (
      np.array([[[ 1.2246468e-16, -1.0000000e+00,  0.0000000e+00],
             [ 1.2246468e-16, -1.0000000e+00,  0.0000000e+00],
             [ 1.2246468e-16, -1.0000000e+00,  0.0000000e+00]],

            [[ 1.0000000e+00,  6.1232340e-17,  0.0000000e+00],
             [ 1.0000000e+00,  6.1232340e-17,  0.0000000e+00],
             [ 1.0000000e+00,  6.1232340e-17,  0.0000000e+00]],

            [[-0.0000000e+00,  1.0000000e+00,  0.0000000e+00],
             [-0.0000000e+00,  1.0000000e+00,  0.0000000e+00],
             [-0.0000000e+00,  1.0000000e+00,  0.0000000e+00]],

            [[-1.0000000e+00,  6.1232340e-17,  0.0000000e+00],
             [-1.0000000e+00,  6.1232340e-17,  0.0000000e+00],
             [-1.0000000e+00,  6.1232340e-17,  0.0000000e+00]],

            [[-1.2246468e-16, -1.0000000e+00,  0.0000000e+00],
             [-1.2246468e-16, -1.0000000e+00,  0.0000000e+00],
             [-1.2246468e-16, -1.0000000e+00,  0.0000000e+00]]]),
    np.array([[[-1.0000000e+00, -1.2246468e-16,  6.1232340e-17],
             [ 0.0000000e+00,  0.0000000e+00,  1.0000000e+00],
             [ 1.0000000e+00,  1.2246468e-16,  6.1232340e-17]],

            [[ 6.1232340e-17, -1.0000000e+00,  6.1232340e-17],
             [-0.0000000e+00,  0.0000000e+00,  1.0000000e+00],
             [-6.1232340e-17,  1.0000000e+00,  6.1232340e-17]],

            [[ 1.0000000e+00,  0.0000000e+00,  6.1232340e-17],
             [-0.0000000e+00, -0.0000000e+00,  1.0000000e+00],
             [-1.0000000e+00, -0.0000000e+00,  6.1232340e-17]],

            [[ 6.1232340e-17,  1.0000000e+00,  6.1232340e-17],
             [-0.0000000e+00, -0.0000000e+00,  1.0000000e+00],
             [-6.1232340e-17, -1.0000000e+00,  6.1232340e-17]],

            [[-1.0000000e+00,  1.2246468e-16,  6.1232340e-17],
             [ 0.0000000e+00, -0.0000000e+00,  1.0000000e+00],
             [ 1.0000000e+00, -1.2246468e-16,  6.1232340e-17]]])
    )

    basis = local_basis_S2(Lon, Lat, deg2rad=True)

    assert np.allclose(basis, basis_expected)

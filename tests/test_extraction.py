import numpy as np
from numbacs.extraction import (ftle_ridge_pts, ftle_ridges, ftle_ordered_ridges,
                                hyperbolic_lcs, hyperbolic_oecs, rotcohvrt)


def test_ftle_ridge_pts(coords_dg,ftle_data,C_eig_data,ridge_pts_data):

    x,y = coords_dg
    _, evecs = C_eig_data
    ridge_pts = ftle_ridge_pts(ftle_data,evecs[:,:,:,1],x,y)

    assert np.allclose(ridge_pts,ridge_pts_data)

def test_ftle_ridges(coords_dg,ftle_data,C_eig_data,ridge_data):

    x,y = coords_dg
    _, evecs = C_eig_data
    ridges = ftle_ridges(ftle_data,evecs[:,:,:,1],x,y)

    assert all([np.allclose(r,rd) for r,rd in zip(ridges,ridge_data)])

def test_ftle_ordered_ridges(coords_dg,ftle_data,C_eig_data,ordered_ridge_data):
    dist_tol = 1e-1
    x,y = coords_dg
    _, evecs = C_eig_data
    ordered_ridges = ftle_ordered_ridges(ftle_data,evecs[:,:,:,1],x,y,dist_tol)

    assert all([np.allclose(r,rd) for r,rd in zip(ordered_ridges,ordered_ridge_data)])

def test_hyperbolic_lcs(coords_dg,C_eig_aux_data,hyp_lcs_data):

    step_size = 1e-2
    steps = 100
    lf = 0.1
    lmin = 0.5
    r = 0.1
    nmax = 1
    dtol = 0.
    nlines = 10
    ep_dist_tol=0.
    lambda_avg_min = 10
    percentile=50
    arclen_flag=False
    x,y = coords_dg
    evals, evecs = C_eig_aux_data
    lcs = hyperbolic_lcs(evals[:,:,1].astype(np.float64), evecs.astype(np.float64), x, y,
                         step_size, steps, lf, lmin, r, nmax,
                         dist_tol=dtol, nlines=nlines,ep_dist_tol=ep_dist_tol,
                         lambda_avg_min = lambda_avg_min,
                         arclen_flag=arclen_flag, percentile=percentile)

    assert all([np.allclose(lcs,lcs_d) for lcs,lcs_d in zip(lcs,hyp_lcs_data)])

def test_hyperbolic_oecs(coords_dg,vel_data,S_eig_data,hyp_oecs_data):

    x,y = coords_dg
    evals, evecs = S_eig_data
    s2 = evals[:,:,1]
    r = .15
    h = 1e-2
    steps = 100
    maxlen = 0.05
    minval = np.percentile(s2,50)
    n = 10

    oecs = hyperbolic_oecs(s2, evecs, x, y, r, h, steps, maxlen, minval,n=n)

    assert all([[np.allclose(oecs,oecs_d) for oecs,oecs_d in zip(foecs, foecs_d)]
                for foecs,foecs_d in zip(oecs,hyp_oecs_data)])

def test_elliptic_lcs(coords_dg,lavd_data,elliptic_lcs_data):

    r = 0.5
    convexity_deficiency = 1e-3
    x,y = coords_dg
    elcs = rotcohvrt(lavd_data,x,y,r,convexity_deficiency=convexity_deficiency)

    assert all([[np.allclose(lcs,lcs_d) for lcs,lcs_d in zip(flcs, flcs_d)]
                for flcs,flcs_d in zip(elcs,elliptic_lcs_data)])

# redundent due to the above function
# def test_elliptic_oecs(coords_dg,ivd_data,elliptic_oecs_data):

#     r = 0.5
#     convexity_deficiency = 1e-3
#     x,y = coords_dg
#     eoecs = rotcohvrt(ivd_data,x,y,r,convexity_deficiency=convexity_deficiency)

#     assert all([[np.allclose(oecs,oecs_d) for oecs,oecs_d in zip(foecs, foecs_d)]
#                 for foecs,foecs_d in zip(eoecs,elliptic_oecs_data)])

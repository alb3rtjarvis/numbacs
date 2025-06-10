import numpy as np
from interpolation.splines import UCGrid
from numbacs.flows import get_callable_scalar_linear
from numbacs.diagnostics import (ftle_grid_2D, C_tensor_2D, C_eig_aux_2D, C_eig_2D,
                                 lavd_grid_2D, ile_2D_func, S_eig_2D_func, S_2D_func,
                                 ile_2D_data, S_eig_2D_data, ivd_grid_2D)

# def evecs_allclose(evecs, evecs_expected, rtol=1e-5, atol=1e-8):
#     """
#     Check if eigenvector angles are close
#     """

#     norms = np.linalg.norm(evecs, axis=-2)
#     norms_expected = np.linalg.norm(evecs_expected, axis=-2)
#     mask = np.logical_and(norms > atol, norms_expected > atol)[:,:,0]
#     evecs = np.divide(evecs[mask,:,:], norms[mask,np.newaxis,:])
#     evecs_expected = np.divide(evecs_expected[mask,:,:], norms_expected[mask,np.newaxis,:])
#     dotprod = np.einsum("...ab,...ab->...b", evecs, evecs_expected)

#     return np.allclose(np.abs(dotprod), 1.0, rtol=rtol, atol=atol)

def evecs_allclose(evecs, evecs_expected, rtol=1e-5, atol=1e-8):
    """
    Check if eigenvector angles are close
    """

    norms = np.linalg.norm(evecs, axis=-2)
    norms_expected = np.linalg.norm(evecs_expected, axis=-2)
    zero_mask = np.logical_and(norms < atol, norms_expected < atol)

    reshaped_norms = np.expand_dims(norms, axis=2)
    reshaped_norms_expected = np.expand_dims(norms_expected, axis=2)



    evecs_n = np.divide(
        evecs, reshaped_norms, out=np.zeros_like(evecs), where=reshaped_norms != 0
    )
    evecs_n_expected = np.divide(
        evecs_expected,
        reshaped_norms_expected,
        out=np.zeros_like(evecs),
        where=reshaped_norms_expected != 0
    )

    dotprod = np.einsum("...ab,...ab->...b", evecs_n, evecs_n_expected)
    parallel = np.isclose(np.abs(dotprod), 1.0, rtol=rtol, atol=atol)

    if not np.all(parallel):
        return False, dotprod
    else:
        return True, np.all(parallel | zero_mask)

def test_ftle_grid_2D(coords_dg, fm_data, ftle_data):

    x,y = coords_dg
    dx = x[1]
    dy = y[1]
    T = 8.
    ftle = ftle_grid_2D(fm_data,T,dx,dy).astype(np.float32)

    assert np.allclose(ftle,ftle_data)


def test_C_tensor_2D(coords_dg, fm_aux_data, C_data):

    x,y = coords_dg
    dx = x[1]
    dy = y[1]
    C = C_tensor_2D(fm_aux_data,dx,dy).astype(np.float32)

    assert np.allclose(C,C_data)

def test_C_eig_aux_2D(coords_dg, fm_aux_data, C_eig_aux_data):

    x,y = coords_dg
    dx = x[1]
    dy = y[1]
    Cvals_expected, Cvecs_expected = C_eig_aux_data
    Cvals, Cvecs = C_eig_aux_2D(fm_aux_data,dx,dy)

    assert np.allclose(Cvals.astype(np.float32),Cvals_expected)
    assert evecs_allclose(Cvecs.astype(np.float32), Cvecs_expected)

def test_C_eig_2D(coords_dg, fm_data, C_eig_data):

    x,y = coords_dg
    dx = x[1]
    dy = y[1]
    Cvals_expected, Cvecs_expected = C_eig_data
    Cvals, Cvecs = C_eig_2D(fm_data,dx,dy)

    assert np.allclose(Cvals.astype(np.float32),Cvals_expected)
    assert evecs_allclose(Cvecs.astype(np.float32), Cvecs_expected)


def test_lavd_grid_2D(coords_dg, fm_n_data, vort_data, lavd_data):

    x,y = coords_dg
    [X,Y] = np.meshgrid(x,y,indexing='ij')
    xrav = X.ravel()
    yrav = Y.ravel()
    T = 8.
    tspan = np.linspace(0.,8.,4)
    grid_vort = UCGrid((0.,8.,4),(0.,2.,21),(0.,1.,11))
    vort_interp = get_callable_scalar_linear(grid_vort,vort_data)
    lavd = lavd_grid_2D(fm_n_data,tspan,T,vort_interp,xrav,yrav).astype(np.float32)

    assert np.allclose(lavd,lavd_data)

def test_ile_2D_func(coords_dg, flow_callable, ile_data):

    vel_func = flow_callable
    x,y = coords_dg
    dx = x[1]
    t0 = 0.

    ile = ile_2D_func(vel_func,x,y,t0=t0,h=dx).astype(np.float32)

    assert np.allclose(ile,ile_data)

def test_S_eig_2D_func(coords_dg, flow_callable, S_eig_func_data):

    x,y = coords_dg
    dx = x[1]
    t0 = 0.
    vel_func = flow_callable
    Svals_expected, Svecs_expected = S_eig_func_data
    Svals, Svecs = S_eig_2D_func(vel_func,x,y,t0=t0,h=dx)
    is_close = evecs_allclose(Svecs.astype(np.float32), Svecs_expected)
    if not is_close[0]:
        dotprod = is_close[1]
        inds = np.argwhere(np.abs(dotprod) < 1)
        print(f"Indices not parallel: {inds}")
        print(f"Vectors computed: {Svecs[tuple(inds.T)]}")
        print(f"Vectors expected: {Svecs_expected[tuple(inds.T)]}")

    assert np.allclose(Svals.astype(np.float32),Svals_expected)
    assert is_close[0]
    # assert evecs_allclose(Svecs.astype(np.float32), Svecs_expected)

def test_S_2D_func(coords_dg, flow_callable, S_data):

    x,y = coords_dg
    dx = x[1]
    t0 = 0.
    vel_func = flow_callable
    S_expected  = S_data
    S = S_2D_func(vel_func,x,y,t0=t0,h=dx).astype(np.float32)

    assert np.allclose(S,S_expected)

def test_ile_2D_data(coords_dg, vel_data, ile_data):

    x,y = coords_dg
    dx = x[1]
    dy = y[1]
    u,v = vel_data
    ile = ile_2D_data(u,v,dx,dy).astype(np.float32)

    assert np.allclose(ile,ile_data)

def test_S_eig_2D_data(coords_dg, vel_data, S_eig_data):

    x,y = coords_dg
    dx = x[1]
    dy = y[1]
    u,v = vel_data
    Svals_expected, Svecs_expected = S_eig_data
    Svals, Svecs = S_eig_2D_data(u,v,dx,dy)

    assert np.allclose(Svals.astype(np.float32),Svals_expected)
    assert evecs_allclose(Svecs.astype(np.float32), Svecs_expected)

def test_ivd_grid_2D(vort_data, ivd_data):

    vort = vort_data[0,:,:]
    vort_avg = np.mean(vort)
    ivd = ivd_grid_2D(vort,vort_avg).astype(np.float32)

    assert np.allclose(ivd,ivd_data)

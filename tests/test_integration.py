import pytest
import numpy as np
from interpolation.splines import UCGrid
from numbacs.flows import get_predefined_flow
from numbacs.integration import (flowmap, flowmap_n,
                                 flowmap_grid_2D,
                                 flowmap_n_grid_2D,
                                 flowmap_aux_grid_2D,
                                 flowmap_composition_initial,
                                 flowmap_composition_step)

funcptr, params, domain = get_predefined_flow('double_gyre')

nx = 21
ny = 11
n = 4

t0 = 0.
T = 8.

@pytest.fixture
def pts_dg(coords_dg):

    x,y = coords_dg
    [X,Y] = np.meshgrid(x,y,indexing='ij')
    pts = np.column_stack((X.ravel(),Y.ravel()))

    return pts

def apply_mask(arr, mask):
    """Apply a mask to arr and return 0.0 where mask is True"""

    arr_masked = arr.copy()
    arr_masked[mask] = 0.0

    return arr_masked

def test_flowmap(pts_dg, fm_data):

    fm = flowmap(funcptr, t0, T, pts_dg, params).reshape(nx,ny,2).astype(np.float32)

    assert np.allclose(fm,fm_data)


def test_flowmap_n(pts_dg, fm_n_data):

    t_eval_expected = params[0]*np.linspace(t0,t0+T,n)
    fm_n, t_eval = flowmap_n(funcptr, t0, T, pts_dg, params, n = 4)
    fm_n = fm_n.reshape(nx,ny,n,2).astype(np.float32)

    assert np.allclose(t_eval_expected,t_eval)
    assert np.allclose(fm_n,fm_n_data)


def test_flowmap_grid_2D(coords_dg, fm_data, mask_dg):

    x,y = coords_dg
    fm = flowmap_grid_2D(funcptr, t0, T, x, y, params).astype(np.float32)

    assert np.allclose(fm, fm_data)

    fm_masked = flowmap_grid_2D(funcptr, t0, T, x, y, params, mask=mask_dg).astype(np.float32)

    assert np.allclose(fm_masked, apply_mask(fm_data, mask_dg))

def test_flowmap_aux_grid_2D(coords_dg, fm_aux_data, mask_dg):

    x,y = coords_dg
    fm_aux = flowmap_aux_grid_2D(funcptr, t0, T, x, y, params).astype(np.float32)

    assert np.allclose(fm_aux,fm_aux_data)

    fm_aux_masked = flowmap_aux_grid_2D(funcptr, t0, T, x, y, params, mask=mask_dg).astype(np.float32)

    assert np.allclose(fm_aux_masked, apply_mask(fm_aux_data, mask_dg))


def test_flowmap_n_grid_2D(coords_dg, fm_n_data, mask_dg):

    x,y = coords_dg
    t_eval_expected = params[0]*np.linspace(t0,t0+T,n)
    fm_n, t_eval = flowmap_n_grid_2D(funcptr, t0, T, x, y, params, n=n)

    assert np.allclose(t_eval_expected,t_eval)
    assert np.allclose(fm_n.astype(np.float32),fm_n_data)

    fm_n_masked, _ = flowmap_n_grid_2D(funcptr, t0, T, x, y, params, n=n, mask=mask_dg)

    assert np.allclose(fm_n_masked.astype(np.float32), apply_mask(fm_n_data, mask_dg))


def test_flowmap_composition_initial(coords_dg, fm_ci_data, fms_ci_data):

    x,y = coords_dg
    nT_expected = 8
    h = 1.
    grid = UCGrid((x[0],x[-1],nx),(y[0],y[-1],ny))
    flowmap0, flowmaps, nT = flowmap_composition_initial(funcptr,t0,T,h,x,y,grid,params)

    assert (np.allclose(flowmap0.astype(np.float32), fm_ci_data)
            & np.allclose(flowmaps.astype(np.float32), fms_ci_data)
            & (nT == nT_expected))


def test_flowmap_composition_step(coords_dg, fms_ci_data, fm_cs_data, fms_cs_data):

    x,y = coords_dg
    h = 1.
    grid = UCGrid((x[0],x[-1],nx),(y[0],y[-1],ny))
    flowmap_k, flowmaps = flowmap_composition_step(fms_ci_data.astype(np.float64),
                                                   funcptr,t0 + T,h,8,x,y,grid,params)

    assert (np.allclose(flowmap_k.astype(np.float32), fm_cs_data)
            & np.allclose(flowmaps.astype(np.float32), fms_cs_data))

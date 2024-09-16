import sys
import pytest
import numpy as np
from math import pi
from numbacs.flows import get_predefined_flow, get_predefined_callable
from numbacs.integration import (flowmap, flowmap_n, flowmap_grid_2D, flowmap_n_grid_2D,
                                 flowmap_aux_grid_2D)

funcptr, params, domain = get_predefined_flow('double_gyre')
dg_func = get_predefined_callable('double_gyre', return_domain=False)

nx = 101
ny = 51
n = 9

x = np.linspace(domain[0][0],domain[0][1],nx)
y = np.linspace(domain[1][0],domain[1][1],ny)
dx = x[1] - x[0]

t0 = 0.
T = 8.


def test_flowmap(fm_data):
    
    [X,Y] = np.meshgrid(x,y,indexing='ij')
    pts = np.column_stack((X.ravel(),Y.ravel()))
    
    fm = flowmap(funcptr, t0, T, pts, params).reshape(nx,ny,2)
    
    assert np.allclose(fm,fm_data)


def test_flowmap_n(fm_n_data):
    
    [X,Y] = np.meshgrid(x,y,indexing='ij')
    pts = np.column_stack((X.ravel(),Y.ravel()))
    
    t_eval_expected = params[0]*np.linspace(t0,t0+T,n)
    fm_n, t_eval = flowmap_n(funcptr, t0, T, pts, params, n = 9)
    fm_n = fm_n.reshape(nx,ny,n,2)
    
    assert np.allclose(t_eval_expected,t_eval)
    assert np.allclose(fm_n,fm_n_data)

    
def test_flowmap_grid_2D(fm_data):
    
    fm = flowmap_grid_2D(funcptr, t0, T, x, y, params)
    
    assert np.allclose(fm,fm_data)
    
    
def test_flowmap_aux_grid_2D(fm_aux_data):
    
    fm_aux = flowmap_aux_grid_2D(funcptr, t0, T, x, y, params)
    
    assert np.allclose(fm_aux,fm_aux_data) 
 
    
def test_flowmap_n_grid_2D(fm_n_data):
    
    t_eval_expected = params[0]*np.linspace(t0,t0+T,n)
    fm_n, t_eval = flowmap_n_grid_2D(funcptr, t0, T, x, y, params, n = n)
    
    assert np.allclose(t_eval_expected,t_eval)
    assert np.allclose(fm_n,fm_n_data)   

import sys
sys.path.append('/home/ajarvis/numbacs_git/src')
import pytest
import numpy as np
from math import pi
from interpolation.splines import UCGrid
from numbacs.flows import get_predefined_flow, get_predefined_callable, get_callable_scalar_linear
from numbacs.diagnostics import (ftle_grid_2D, C_tensor_2D, C_eig_aux_2D, C_eig_2D, ftle_from_eig,
                                 lavd_grid_2D, ile_2D_func, S_eig_2D_func, S_2D_func, ile_2D_data,
                                 S_eig_2D_data, ivd_grid_2D)

# funcptr, params = get_predefined_flow('double_gyre')
# dg_func = get_predefined_callable('double_gyre', return_domain=False)

# nx = 101
# ny = 51
# n = 9

# t0 = 0.
# T = 8.

# @pytest.fixture
# def fm_data():

#     return np.load("./testing_data/fm.npy")

# @pytest.fixture
# def fm_n_data():

#     return np.load("./testing_data/fm_n.npy")

# @pytest.fixture
# def fm_aux_data():

#     return np.load("./testing_data/fm_aux.npy")

# @pytest.fixture
# def ftle_data():

#     return np.load("./testing_data/ftle.npy")

# @pytest.fixture
# def C_data():

#     return np.load("./testing_data/C.npy")

# @pytest.fixture
# def C_eig_aux_data():

#     return np.load("./testing_data/Cvals_aux.npy"), np.load("./testing_data/Cvecs_aux.npy")

# @pytest.fixture
# def C_eig_data():

#     return np.load("./testing_data/Cvals.npy"), np.load("./testing_data/Cvecs.npy")

# @pytest.fixture
# def lavd_data():

#     return np.load("./testing_data/lavd.npy")

# @pytest.fixture
# def ile_data():

#     return np.load("./testing_data/ile.npy")

# @pytest.fixture
# def S_data():

#     return np.load("./testing_data/S.npy")

# @pytest.fixture
# def S_eig_func_data():

#     return np.load("./testing_data/Svals.npy"), np.load("./testing_data/Svecs.npy")

# @pytest.fixture
# def S_eig_data():

#     return np.load("./testing_data/Svals_d.npy"), np.load("./testing_data/Svecs_d.npy")

# @pytest.fixture
# def ivd_data():

#     return np.load("./testing_data/ivd.npy")

# @pytest.fixture
# def vort_data():

#     return np.load("./testing_data/vort.npy")

# @pytest.fixture
# def vel_data():

#     return np.load("./testing_data/u.npy"), np.load("./testing_data/v.npy")

# @pytest.fixture
# def flow():
    
#     return get_predefined_flow('double_gyre', return_domain=False)

# @pytest.fixture
# def flow_callable():
    
#     return get_predefined_callable('double_gyre', return_domain=False)

# @pytest.fixture
# def coords_dg():
#     x = np.linspace(0,2,101)
#     y = np.linspace(0,1,51)
    
#     return x,y

def test_ftle_grid_2D(fm_data,ftle_data,coords_dg):
    
    x,y = coords_dg
    dx = x[1]
    dy = y[1]
    T = 8.
    ftle = ftle_grid_2D(fm_data,T,dx,dy)
        
    assert np.allclose(ftle,ftle_data)


def test_C_tensor_2D(fm_aux_data,coords_dg,C_data):
    
    x,y = coords_dg
    dx = x[1]
    dy = y[1]
    C = C_tensor_2D(fm_aux_data,dx,dy)
        
    assert np.allclose(C,C_data)
    
def test_C_eig_aux_2D(fm_aux_data,coords_dg,C_eig_aux_data):
    
    x,y = coords_dg
    dx = x[1]
    dy = y[1]
    Cvals_expected, Cvecs_expected = C_eig_aux_data
    Cvals, Cvecs = C_eig_aux_2D(fm_aux_data,dx,dy)
        
    assert np.allclose(Cvals,Cvals_expected)
    assert np.allclose(Cvecs,Cvecs_expected)
    
def test_C_eig_2D(fm_data,coords_dg,C_eig_data):
    
    x,y = coords_dg
    dx = x[1]
    dy = y[1]
    Cvals_expected, Cvecs_expected = C_eig_data
    Cvals, Cvecs = C_eig_2D(fm_data,dx,dy)
        
    assert np.allclose(Cvals,Cvals_expected)
    assert np.allclose(Cvecs,Cvecs_expected)    

    
def test_lavd_grid_2D(fm_n_data,coords_dg,vort_data,lavd_data):
    
    x,y = coords_dg
    [X,Y] = np.meshgrid(x,y,indexing='ij')
    xrav = X.ravel()
    yrav = Y.ravel()
    T = 8.
    tspan = np.linspace(0.,8.,9)
    grid_vort = UCGrid((0.,8.,9),(0.,2.,101),(0.,1.,51))
    vort_interp = get_callable_scalar_linear(grid_vort,vort_data)
    lavd = lavd_grid_2D(fm_n_data,tspan,T,vort_interp,xrav,yrav)
        
    assert np.allclose(lavd,lavd_data)
    
def test_ile_2D_func(ile_data,coords_dg):
    
    x,y = coords_dg
    dx = x[1]
    t0 = 0.
    vel_func = get_predefined_callable('double_gyre', return_domain=False)
    ile = ile_2D_func(vel_func,x,y,t0=t0,h=dx)
        
    assert np.allclose(ile,ile_data)    

def test_S_eig_2D_func(flow_callable,coords_dg,S_eig_func_data):
    
    x,y = coords_dg
    dx = x[1]
    t0 = 0.
    vel_func = flow_callable
    Svals_expected, Svecs_expected = S_eig_func_data
    Svals, Svecs = S_eig_2D_func(vel_func,x,y,t0=t0,h=dx)
        
    assert np.allclose(Svals,Svals_expected)
    assert np.allclose(Svecs,Svecs_expected)
    
def test_S_2D_func(flow_callable,coords_dg,S_data):
    
    x,y = coords_dg
    dx = x[1]
    t0 = 0.
    vel_func = flow_callable
    S_expected  = S_data
    S = S_2D_func(vel_func,x,y,t0=t0,h=dx)
        
    assert np.allclose(S,S_expected)     
    
def test_ile_2D_data(vel_data,ile_data,coords_dg):
    
    x,y = coords_dg
    dx = x[1]
    dy = y[1]
    u,v = vel_data
    ile = ile_2D_data(u,v,dx,dy)
        
    assert np.allclose(ile,ile_data)  
    
def test_S_eig_2D_data(vel_data,coords_dg,S_eig_data):
    
    x,y = coords_dg
    dx = x[1]
    dy = y[1]
    u,v = vel_data
    Svals_expected, Svecs_expected = S_eig_data
    Svals, Svecs = S_eig_2D_data(u,v,dx,dy)
        
    assert np.allclose(Svals,Svals_expected)
    assert np.allclose(Svecs,Svecs_expected)
    
def test_ivd_grid_2D(vort_data,ivd_data):
    
    vort = vort_data[0,:,:]
    vort_avg = np.mean(vort)
    ivd = ivd_grid_2D(vort,vort_avg)
        
    assert np.allclose(ivd,ivd_data)    

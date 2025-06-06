#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NumbaCS vs SciPy/NumPy -- QGE
=============================

Compare run times for flow map and FTLE between NumbaCS and
a pure SciPy/NumPy implementation for the QGE.
"""

# Author: ajarvis
# Hardware: Intel(R) Core(TM) i7-3770K CPU @ 3.50GHz, Cores = 4, Threads = 8

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from numbacs.integration import flowmap_grid_2D
from numbacs.flows import get_interp_arrays_2D, get_flow_2D
from numbacs.diagnostics import ftle_grid_2D
import time
from math import copysign, log
from multiprocessing import Pool
from functools import partial

# %%
# Get flow data and create NumbaCS flow
# -------------------------------------
# Load velocity data, set up domain, set the integration span and direction, create
# interpolant of velocity data and retrieve necessary arrays.

# load in qge velocity data
u = np.load('../data/qge/qge_u.npy')
v = np.load('../data/qge/qge_v.npy')

# set up domain
nt,nx,ny = u.shape
x = np.linspace(0,1,nx)
y = np.linspace(0,2,ny)
t = np.linspace(0,1,nt)
dx = x[1]-x[0]
dy = y[1]-y[0]

# use reduced domain or scipy will take much too long
s = 4
x = x[::s]
y = y[::s]
t = t[::s]
u = u[::s,::s,::s]
v = v[::s,::s,::s]

nx = len(x)
ny = len(y)

# set integration span and integration direction
t0 = 0.0
T = 0.1
params = np.array([copysign(1,T)])  # important this is an array of type float

# get interpolant arrays of velocity field
grid_vel, C_eval_u, C_eval_v = get_interp_arrays_2D(t, x, y, u, v)

# get flow to be integrated
funcptr = get_flow_2D(grid_vel, C_eval_u, C_eval_v, extrap_mode='linear')


# %%
# Scipy interpolant and ODE function
# ----------------------------------
# Create interpolant and function for SciPy ode solver.


ui = RegularGridInterpolator((t,x,y), u, method= 'linear', bounds_error=False, fill_value=0.0)
vi = RegularGridInterpolator((t,x,y), v, method= 'linear', bounds_error=False, fill_value=0.0)

def odeint_fun(yy,tt):

    pt = np.array([tt,yy[0],yy[1]])

    return ui(pt)[0],vi(pt)[0]


# %%
# SciPy flow map and FTLE functions
# ---------------------------------
# Create functions to compute flow maps and FTLE using standard SciPy/Numpy methods.
# Uses scipy.integrate.odeint (implements LSODA method) for particle integration.
# The scipy.integrate.solve_ivp function is newer and allows the use of other solvers
# but odeint is faster even when solve_ivp uses LSODA as its method.

tspan = np.array([t0,t0+T])

def scipy_odeint_flowmap_par(t0,y0):
    tspan = np.array([t0,t0+T])
    sol = odeint(odeint_fun, y0, tspan, rtol=1e-6, atol=1e-8)
    flowmap = sol[-1,:]

    return flowmap

def numpy_ftle_par(fm,inds):

    i,j = inds
    absT = abs(T)
    dxdx = (fm[i+1,j,0] - fm[i-1,j,0])/(2*dx)
    dxdy = (fm[i,j+1,0] - fm[i,j-1,0])/(2*dy)
    dydx = (fm[i+1,j,1] - fm[i-1,j,1])/(2*dx)
    dydy = (fm[i,j+1,1] - fm[i,j-1,1])/(2*dy)

    off_diagonal = dxdx*dxdy + dydx*dydy
    C = np.array([[dxdx**2 + dydx**2, off_diagonal],
                   [off_diagonal, dxdy**2 + dydy**2]])

    max_eig = np.linalg.eigvalsh(C)[-1]
    if max_eig > 1:
        ftle = 1/(2*absT)*log(max_eig)
    else:
        ftle = 0

    return ftle

# %%
# Compute SciPy/Numpy flow map, FTLE
# ----------------------------------
# Compute flowmap, FTLE, and calculate run times for the SciPy/NumPy implementation.
# For this problem on this hardware, computing flow map and FTLE parallel in space
# (as opposed to parallel in time) was the faster implementation.

# set initial conditions
n = 2
t0span = np.linspace(0,0.1,n)
[X,Y] = np.meshgrid(x,y,indexing='ij')
Y0 = np.column_stack((X.ravel(),Y.ravel()))
sftle = np.zeros((n,nx-2,ny-2),np.float64)

# set parallel pool to use maximum number of threads for this hardware,
# open pool
num_threads = 8
pl = Pool(num_threads)

# create inds to pass to ftle function
xinds = np.arange(1,nx-1)
yinds = np.arange(1,ny-1)
[I,J] = np.meshgrid(xinds,yinds,indexing='ij')
inds = np.column_stack((I.ravel(),J.ravel()))

# compute flowmap and ftle parallel in space
sfmtt = 0
sftt = 0

for k,t0 in enumerate(t0span):
    ks = time.perf_counter()
    func = partial(scipy_odeint_flowmap_par,t0)
    res = np.array(pl.map(func,Y0)).reshape(nx,ny,2)
    kf = time.perf_counter()
    sfmtt += kf - ks

    fks = time.perf_counter()
    func2 = partial(numpy_ftle_par,res)
    sftle[k,:,:] = np.array(pl.map(func2, inds)).reshape(nx-2,ny-2)
    fkf = time.perf_counter()
    sftt += fkf - fks

pl.close()
pl.terminate()

print("SciPy/NumPy flowmap and FTLE took "
      + "{:.5f} seconds for {} iterates".format(sfmtt + sftt, n))
print("Mean time for SciPy/NumPy flowmap and FTLE -- "
      + "{:.5f} seconds\n".format((sfmtt + sftt)/n))
print("Scipy flowmap took {:.5} seconds for {:1d} iterates".format(sfmtt,n))
print("Mean time for Scipy flowmap -- {:.5} seconds\n".format(sfmtt/n))
print("NumPy ftle took {:.5} seconds for {:1d} iterates".format(sftt,n))
print("Mean time for NumPy ftle -- {:.5} seconds\n".format(sftt/n))

# %%
# Compute NumbaCS flow map, FTLE
# --------------------------------
# Compute flowmap, FTLE, and calculate run times for the NumbaCS implementation.
# For this problem on this hardware, computing flow map and FTLE parallel in space
# (as opposed to parallel in time) was the faster implementation.

ftle = np.zeros((n,nx,ny),np.float64)

# first call and record warmup times
wfm = time.perf_counter()
flowmap_wu = flowmap_grid_2D(funcptr,t0,T,x,y,params)
wu_fm = time.perf_counter() - wfm
print("Flowmap with warm-up took {:.5f} seconds".format(wu_fm))

wf = time.perf_counter()
ftle[0,:,:] = ftle_grid_2D(flowmap_wu,T,dx,dy)
wu_f = time.perf_counter() - wf
print("FTLE with warm-up took {:.5f} seconds".format(wu_f))

# initialize runtime counters
fmtt = wu_fm
ftt = wu_f

# loop over initial times, compute flowmap and ftle
for k, t0 in enumerate(t0span[1:]):
    ks = time.perf_counter()
    flowmap = flowmap_grid_2D(funcptr,t0,T,x,y,params)
    kf = time.perf_counter()
    fmtt += kf-ks

    fks = time.perf_counter()
    ftle[k,:,:] = ftle_grid_2D(flowmap,T,dx,dy)
    fkf = time.perf_counter()
    ftt += fkf-fks

print("NumbaCS flowmap and FTLE took "
      + "{:.5f} for {:1d} iterates".format(fmtt+ftt,n))
print("Mean time for flowmap and FTLE -- {:.5f} seconds (w/ warmup)".format((fmtt+ftt)/n))
print("Mean time for flowmap and FTLE -- "
      + "{:.5f} seconds (w/o warmup)\n".format((fmtt-wu_fm+ftt-wu_f)/(n-1)))
print("NumbaCS flowmap_grid_2D took {:.5f} seconds for {:1d} iterates".format(fmtt,n))
print("Mean time for flowmap_grid_2D -- {:.5f} seconds (w/ warmup)".format(fmtt/n))
print("Mean time for flowmap_grid_2D -- "
      + "{:.5f} seconds (w/o warmup)\n".format((fmtt-wu_fm)/(n-1)))
print("NumbaCS ftle_grid_2D took {:.5f} seconds for {:1d} iterates".format(ftt,n))
print("Mean time for ftle_grid_2D -- {:.5f} seconds (w/ warmup)".format(ftt/n))
print("Mean time for ftle_grid_2D -- {:.5f} seconds (w/o warmup)".format((ftt-wu_f)/(n-1)))

# %%
# Compare timings
# ---------------
# Compare timings and quantify speed-up. The second and third columns quantify the
# speed-up gained using NumbaCS. The second column includes warm-up time, the speed-up
# would increase as *n* grows larger. The third column ignores the warm-up time
# and quantifies the speed-up as *n* goes to infinity and the warm-up time becomes
# negligible. This represents the theoretical speed-up.


stt = sfmtt + sftt
ntt = fmtt + ftt

stpi = (sfmtt + sftt)/n
ntpi = (ntt - wu_fm - wu_f)/(n-1)

d1 = 5
d2 = 2
data = [[round(stt,d1),'--','--'],
        [round(ntt,d1),round(stt/ntt,d2),round(stpi/ntpi,d2)]]

times = ["total time (n={})".format(n),"speedup","speedup (n->inf)"]
methods = ["SciPy/NumPy","NumbaCS"]

format_row = "{:>25}"*(len(data[0]) + 1)

print(format_row.format("", *times))

for name, vals in zip(methods,data):
    print(format_row.format(name,*vals))

# %%
#
#
# .. note::
#
#    The SciPy interpolation package creates a bottleneck when used to solve odes and has
#    a large effect on the overall runtime. For this reason, we only run for 2
#    iterates or the code would take much too long. As *n* increaes, the speed-up
#    would increase quite quickly as the warm-up time of the NumbaCS implementation
#    becomes less significant. Regardless, the NumbaCS implementation achieves a
#    drastic speed-up when used on numerical velocity data. This is largely achieved
#    by the numbalsoda and interpolation packages, both of which utilize Numba.

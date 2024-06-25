# NumbaCS

NumbaCS (Numba Coherent Structures) is a Python package for performing coherent structure analysis in a user-friendly and efficient manner. It implements methods we will refer to as "coherent structure methods" which is being used as an umbrella term for any method that can be used to infer or extract Lagrangian and Eulerian coherent structures. These are tools for performing analysis of time-dependent dynamical systems, mainly with a focus on fluid flows. While this package can be used with any (incompressible) flow, methods are provided which make it simple for the user to take advantage of these tools for large-scale geophysical flows.

NumbaCS is built on top of three main packages: [Numba](https://numba.pydata.org), [numbalsoda](https://github.com/Nicholaswogan/numbalsoda), and [interpolation](https://www.econforge.org/interpolation.py/). Numba is a compiler for Python array and numerical functions that generates optimized machine code just-in-time to significantly speed up numerical operations in Python. Numbalsoda is a Python wrapper to ode solvers in both C++ (LSODA) and FORTRAN (DOP853) that is compatible with Numba (standard Python ode solvers cannot be used within Numba functions), speeding up the most computationally expensive piece of Lagrangian coherent structure methods, particle integration. The interpolation package is a Numba-compatible package which optimizes interpolation in Python. It is used here to generate interpolants of numerical velocity fields which can be fed into solvers from the numbalsoda package. All of this interfacing is done behind the scenes through modules which the user can call in a straightforward manner.

## Features

NumbaCS currently implements the following features:

`particle integration` -- any dimension

Diagnostics:

`FTLE` -- any dimension (>=2)

`iLE` -- 2D

`LAVD` -- 2D

`IVD` -- any dimension (>=2)

`DOA` -- any dimension (>=2)

Feature extraction:

`LCS` -- 2D, hyperbolic

`OECS` -- 2D, hyperbolic

`FTLE ridges` -- 2D

`iLE ridges` -- 2D

Features coming shortly:

`DBS` -- 2D, isotropic and anisotropic diffusion

`LCS` -- 2D, elliptic

`OECS` -- 2D, elliptic




## Installation

Package is currently under construction, conda installation coming soon...

## Basic usage

### Predefined flow

```python
import numpy as np
from math import copysign
from numbacs.flows import get_predefined_flow
from numbacs.integration import flowmap_grid_2D
from numbacs.diagnostics import ftle_grid_2D

# set integration span and integration direction
t0 = 0.
T = -10.
int_direction = copysign(1,T)

# get ode to be used by 'flowmap_grid_2D'
funcptr, params, domain = get_predefined_flow('double_gyre', int_direction = int_direction)

# set up domain
nx,ny = 401,201
x = np.linspace(domain[0][0],domain[0][1],nx)
y = np.linspace(domain[1][0],domain[1][1],ny)
dx = abs(x[1]-x[0])
dy = abs(y[1]-y[0])

# computes final position of particle trajectories over grid
flowmap = flowmap_grid_2D(funcptr, t0, T, x, y, params)

# compute FTLE over grid
ftle = ftle_grid_2D(flowmap,T,dx,dy)
```

### Numerical velocity data

```python
import numpy as np
from math import copysign
from numbacs.flows import get_interp_arrays_2D, get_flow_2D
from numbacs.integration import flowmap_grid_2D
from numbacs.diagnostics import ftle_grid_2D

# given you have data in current directory,
# load data, set integration span and direction
t = np.load('./t.npy')
x = np.load('./x.npy')
y = np.load('./y.npy')
U = np.load('./u.npy')
V = np.load('./v.npy')
dx = abs(x[1]-x[0])
dy = abs(y[1]-y[0])
t0 = 0.
T = -10.
params = np.array([copysign(1,T)])

# get ode to be used by 'flowmap_grid_2D'
grid_vel, C_eval_u, C_eval_v = get_interp_arrays_2D(t,x,y,U,V)
funcptr = get_flow_2D(grid_vel, C_eval_u, C_eval_v)

# computes final position of particle trajectories over grid
flowmap = flowmap_grid_2D(funcptr, t0, T, x, y, params)

# compute FTLE over grid
ftle = ftle_grid_2D(flowmap,T,dx,dy)
```
Thorough documentation coming soon...

## Roadmap

Future releases aim to extend certain methods to higher dimensions, implement new features that should be straightforward within this framework (shape coherent sets, etc.), and further streamline and optimize the process for large-scale geophysical applications. 

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

# NumbaCS

**Documentation:** [https://numbacs.readthedocs.io/](https://numbacs.readthedocs.io/en/latest/)

**Source Code (MPL-2.0):** [https://github.com/alb3rtjarvis/numbacs](https://github.com/alb3rtjarvis/numbacs)

NumbaCS (Numba Coherent Structures) is a Python package for performing coherent structure analysis in a user-friendly and efficient manner. It implements methods referred to as "coherent structure methods", an umbrella term for any method that can be used to infer or extract Lagrangian and Eulerian coherent structures. These are tools for performing analysis of time-dependent dynamical systems, mainly with a focus on material transport in fluid flows. While this package can be used with any (incompressible) flow, methods are provided which make it simple for the user to take advantage of these tools for large-scale geophysical flows.

## Features

NumbaCS currently implements the following features:

### Diagnostics:

* **FTLE** -- finite time Lyapunov exponent
* **iLE** -- instantaneous Lyapunov exponent
* **LAVD** -- Lagrangian averaged vorticity deviation
* **IVD** -- instantaneous vorticity deviation

### Feature extraction:

* **LCS** -- hyperbolic and elliptic Lagrangian coherent structures
* **OECS** -- hyperbolic and elliptic objective Eulerian coherent structures
* **Ridge extraction** -- FTLE and iLE ridge extraction

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
int_direction = copysign(1, T)

# get ode to be used by 'flowmap_grid_2D'
funcptr, params, domain = get_predefined_flow('double_gyre', int_direction = int_direction)

# set up domain
nx,ny = 401,201
x = np.linspace(domain[0][0], domain[0][1], nx)
y = np.linspace(domain[1][0], domain[1][1], ny)
dx = abs(x[1] - x[0])
dy = abs(y[1] - y[0])

# computes final position of particle trajectories over grid
flowmap = flowmap_grid_2D(funcptr, t0, T, x, y, params)

# compute FTLE over grid
ftle = ftle_grid_2D(flowmap, T, dx, dy)
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
dx = abs(x[1] - x[0])
dy = abs(y[1] - y[0])
t0 = 0.
T = -10.
params = np.array([copysign(1, T)])

# get ode to be used by 'flowmap_grid_2D'
grid_vel, C_eval_u, C_eval_v = get_interp_arrays_2D(t, x, y, U, V)
funcptr = get_flow_2D(grid_vel, C_eval_u, C_eval_v)

# computes final position of particle trajectories over grid
flowmap = flowmap_grid_2D(funcptr, t0, T, x, y, params)

# compute FTLE over grid
ftle = ftle_grid_2D(flowmap, T, dx, dy)
```

## Key dependencies

NumbaCS is built on top of three main packages: [Numba](https://numba.pydata.org), [numbalsoda](https://github.com/Nicholaswogan/numbalsoda), and [interpolation](https://www.econforge.org/interpolation.py/). Numba is a JIT-compiler for Python array and numerical functions that generates optimized machine code "just-in-time" (using the LLVM compiler library) to significantly speed up numerical operations in Python. Numbalsoda is a Python wrapper to ODE solvers in both C++ (LSODA) and FORTRAN (DOP853) that is compatible with Numba (standard Python ODE solvers cannot be used within Numba functions) and bypasses the Python interpreter, speeding up the most computationally expensive piece of Lagrangian coherent structure methods, particle integration. The interpolation package is a Numba-compatible package which optimizes interpolation in Python. It is used in NumbaCS to generate JIT-compiled interpolant functions of numerical velocity fields which can be fed into solvers from the numbalsoda package, resulting in efficient particle integration for flows defined by numerical velocity data. All of this interfacing, which is done behind the scenes through modules the user can call in a straightforward manner, is how NumbaCS achieves impressive speeds while maintaining a relatively simple user experience. We are grateful to the creators, maintainers, and contributors of each of these packages, as well as the other dependencies which NumbaCS relies on ([NumPy](https://numpy.org/), [SciPy](https://scipy.org/), and [ContourPy](https://contourpy.readthedocs.io/en/v1.3.0/)).

## Roadmap

Future releases aim to extend certain methods to higher dimensions, implement new features that should be straightforward within this framework (shape coherent sets, lobe dynamics, etc.), and further streamline and optimize the process for large-scale geophysical applications. 

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Similar software

[Lagrangian](https://lagrangian.readthedocs.io/en/latest/index.html) -- Python
package for computing FSLE, FTLE, and eigenvectors of Cauchy-Green tensor with a
focus on geophysical flows. Largely written in C++ with pybind11 used for
binding, producing fast code. Particle integration is performed by
4th order Runge-Kutta method (RK4).

[Dynlab](https://github.com/hokiepete/dynlab) --  Object oriented Python package
which computes Lagrangian and Eulerian diagnostics along with ridge extraction.
Provides a large collection of predefined flows and is very user friendly.
Particle integration performed by scipy.integrate.odeint (LSODA).

[TBarrier](https://github.com/haller-group/TBarrier) -- Collection of Jupyter
notebooks accompanying the book *Transport Barriers and Coherent Structures
in Flow Data -- Advective, Diffusive, Stochastic, and Active methods by George
Haller*. Python code which implements a variety of Lagrangian and Eulerian
diagnostics and extraction methods for a variety of different transport settings
(NumbaCS currently only implements purely advective methods). Particle
integration performed by 4th order Runge-Kutta method (RK4).

[Newman](https://github.com/RossDynamics/Newmanv3.1) -- Fast C++ code for
computing FTLE which works with geophysical flows and storm tracking. Various
methods for particle integration, both fixed and adaptive step-size methods. No
longer maintained.

[Aquila-LCS](https://github.com/ChristianLagares/Aquila-LCS) -- Python code
designed to compute FTLE for high-speed turbulent boundary layers in 3D.
Utilizes Numba to implement GPU and CPU versions of the code for fast execution
times. Particle integration performed by Euler method.

[CoherentStructures.jl](https://coherentstructures.github.io/CoherentStructures.jl/stable/) 
-- Julia toolbox for computing LCS/FTCS in aperiodic flows. Implements elliptic
LCS methods, FEM-based methods (FEM approximation of dynamic Laplacian for FTCS
extraction), and Graph Laplacian-based methods (spectral clustering and
diffusion maps for coherent sets). Makes use of DifferentialEquations.jl for
particle integration, a very advanced and efficient suite of DE solvers in
Julia.

[LCS Tool](https://github.com/haller-group/LCStool) MATLAB code used to compute
Elliptic LCS, Hyperbolic LCS, and FTLE. Particle integration performed by
MATLAB's ode45 function (based off of RK5(4) due to Dormand and Prince).

#!/usr/bin/env python3
"""

Double gyre FTLE
================

Compute the FTLE field for the double gyre.
"""

# Author: ajarvis

import numpy as np
from math import copysign
import matplotlib.pyplot as plt
from numbacs.flows import get_predefined_flow
from numbacs.integration import flowmap_grid_2D
from numbacs.diagnostics import ftle_grid_2D
# %%
# Get flow
# --------------
# Set the integration span and direction, retrieve the flow, and set up domain.

# set initial time, integration time, and integration direction
t0 = 0.
T = -10.
int_direction = copysign(1,T)

# retrieve function pointer, parameters, and domain for double gyre flow.
funcptr, params, domain = get_predefined_flow('double_gyre', int_direction = int_direction)

# set up domain
nx,ny = 401,201
x = np.linspace(domain[0][0],domain[0][1],nx)
y = np.linspace(domain[1][0],domain[1][1],ny)
dx = x[1]-x[0]
dy = y[1]-y[0]

# %%
# Integrate
# ---------
# Integrate grid of particles and return final positions.
flowmap = flowmap_grid_2D(funcptr, t0, T, x, y, params)

# %%
# FTLE
# ----
# Compute FTLE field from final particle positions.
ftle = ftle_grid_2D(flowmap,T,dx,dy)

# %%
# Plot
# ----
# Plot the results.
fig,ax = plt.subplots(dpi=200)
ax.contourf(x,y,ftle.T,levels=80)
ax.set_aspect('equal')
plt.show()

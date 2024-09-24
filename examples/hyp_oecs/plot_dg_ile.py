#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Double gyre iLE
===============

Compute the iLE field for the double gyre.
"""

# Author: ajarvis

import numpy as np
import matplotlib.pyplot as plt
from numbacs.flows import get_predefined_callable
from numbacs.diagnostics import ile_2D_func
# %%
# Get flow
# --------------
# Set the initial time, retrieve a jit-callable for the flow, and set up domain.

# set time at which to compute iLE
t0 = 0.

# retrieve function pointer, parameters, and domain for double gyre flow.
vel_func, domain = get_predefined_callable('double_gyre')

# set up domain
nx,ny = 401,201
x = np.linspace(domain[0][0],domain[0][1],nx)
y = np.linspace(domain[1][0],domain[1][1],ny)
dx = x[1]-x[0]
dy = y[1]-y[0]

# %%
# iLE
# ---------
# Compute iLE from velocity function at t = t0.
ile = ile_2D_func(vel_func, x, y, t0 = t0)

# %%
# Plot
# ----
# Plot the results.
fig,ax = plt.subplots(dpi=200)
ax.contourf(x,y,ile.T,levels=80)
ax.set_aspect('equal')
plt.show()

#!/usr/bin/env python3
"""
Bickley jet FTLE ridges
=======================

Compute the FTLE field and ridges for the bickley jet.
"""

# Author: ajarvis

import numpy as np
from math import copysign
import matplotlib.pyplot as plt
from numbacs.flows import get_predefined_flow
from numbacs.integration import flowmap_grid_2D
from numbacs.diagnostics import ftle_from_eig, C_eig_2D
from numbacs.extraction import ftle_ordered_ridges
# %%
# Get flow
# --------------
# Set the integration span and direction, retrieve the flow, and set up domain.

# set initial time, integration time, and integration direction
t0 = 0.0
T = 6.0
int_direction = copysign(1, T)

# retrieve function pointer, parameters, and domain for bickley jet flow.
funcptr, params, domain = get_predefined_flow("bickley_jet", int_direction=int_direction)

# set up domain
dx = 0.05
dy = 0.05
x = np.arange(domain[0][0], domain[0][1] + dx, dx)
y = np.arange(domain[1][0], domain[1][1] + dy, dy)
dx = x[1] - x[0]
dy = y[1] - y[0]

# %%
# Integrate
# ---------
# Integrate grid of particles and return final positions.
flowmap = flowmap_grid_2D(funcptr, t0, T, x, y, params)

# %%
# CG eigenvalues, eigenvectors, and FTLE
# ----------------------------------------------
# Compute eigenvalues/vectors of CG tensor from final particle positions and compute FTLE.

# compute eigenvalues/vectors of Cauchy Green tensor
eigvals, eigvecs = C_eig_2D(flowmap, dx, dy)
eigval_max = eigvals[:, :, 1]
eigvec_max = eigvecs[:, :, :, 1]

# compute FTLE from max eigenvalue
ftle = ftle_from_eig(eigval_max, T)

# %%
# Ridge extraction
# ----------------
# Compute ordered FTLE ridges.

# set parameters for ridge function
percentile = 90
sdd_thresh = 10.0

# identify ridge points, link points in each ridge in an ordered manner,
# connect close enough ridges
dist_tol = 1e-1
ridge_curves = ftle_ordered_ridges(
    ftle, eigvec_max, x, y, dist_tol, percentile=percentile, sdd_thresh=sdd_thresh, min_ridge_pts=25
)
# %%
# Plot
# ----
# Plot the results.
fig, ax = plt.subplots(dpi=200)
ax.contourf(x, y, ftle.T, levels=80)
for rc in ridge_curves:
    ax.plot(rc[:, 0], rc[:, 1], lw=1.5)
ax.set_aspect("equal")
plt.show()

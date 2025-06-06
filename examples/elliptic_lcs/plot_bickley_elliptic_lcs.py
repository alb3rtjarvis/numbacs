#!/usr/bin/env python3
"""
Bickley jet Elliptic LCS
========================

Compute the LAVD-based elliptic lcs for the bickley jet.
"""

# Author: ajarvis

import numpy as np
from math import copysign, pi
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numbacs.flows import (
    get_predefined_flow,
    get_predefined_callable,
    get_interp_arrays_scalar,
    get_callable_scalar,
)
from numbacs.integration import flowmap_n_grid_2D, flowmap_n
from numbacs.diagnostics import lavd_grid_2D
from numbacs.extraction import rotcohvrt
from numbacs.utils import curl_func_tspan, gen_filled_circ, interp_curve, pts_in_poly_mask
# %%
# Get flow
# --------------
# Set the integration span and direction, retrieve the flow, jit-callable for velocity
# field, and set up domain.

# set initial time, integration time, integration direction,
# and integration span (units of days)
t0 = 0.0
T = 40.0
int_direction = copysign(1, T)
n = 801

# retrieve function pointer and parameters for bickley jet flow.
funcptr, params = get_predefined_flow(
    "bickley_jet", int_direction=int_direction, return_domain=False
)

# retrieve callable to compute vorticity
vel_spline = get_predefined_callable("bickley_jet", return_domain=False)

# set up larger domain to capture all elliptic lcs
domain = ((-2.0, 6.371 * pi + 2.0), (-3.0, 3.0))
dx = 0.05
dy = 0.05
x = np.arange(domain[0][0], domain[0][1] + dx, dx)
y = np.arange(domain[1][0], domain[1][1] + dy, dy)
nx = len(x)
ny = len(y)

# %%
# Integrate
# ---------
# Integrate grid of particles and return positions at n times between [t0,t0+T] (inclusive).
flowmap, tspan = flowmap_n_grid_2D(funcptr, t0, T, x, y, params, n=n)

# %%
# Vorticity
# ---------
# Copmute vorticity on the grid and over the times for which the flowmap was returned.

# compute vorticity and create interpolant for it
vort = curl_func_tspan(vel_spline, tspan, x, y, h=1e-3)
grid_domain, C_vort = get_interp_arrays_scalar(tspan, x, y, vort)
vort_spline = get_callable_scalar(grid_domain, C_vort)
# %%
# LAVD
# ----
# Compute LAVD from vorticity and particle positions.

# need to pass raveled arrays into lavd_grid_2D
X, Y = np.meshgrid(x, y, indexing="ij")
xrav = X.ravel()
yrav = Y.ravel()

# since bickley jet is periodic in x-direction, need to pass its period
# into lavd_grid_2D
period_x = 6.371 * pi

# compute lavd
lavd = lavd_grid_2D(flowmap, tspan, T, vort_spline, xrav, yrav, period_x=period_x)

# %%
# LAVD-based elliptic LCS
# -----------------------
# Compute elliptic LCS from LAVD.

# set parameters and compute lavd-based elliptic lcs
r = 2.5
convexity_deficiency = 4e-3
elcs = rotcohvrt(lavd, x, y, r, convexity_deficiency=convexity_deficiency)
# %%
# Plot
# ----
# Plot the elliptic LCS over the LAVD field.

# sphinx_gallery_thumbnail_number = 1
fig, ax = plt.subplots(dpi=200)
ax.contourf(x, y, lavd.T, levels=80)
ax.set_aspect("equal")
for rcv, c in elcs:
    ax.plot(rcv[:, 0], rcv[:, 1], lw=1.5)
    ax.scatter(c[0], c[1], 1.5)
plt.show()

# %%
# Advect elliptic LCS
# -------------------
# Advect an elliptic LCS and nearby filled circle of points to demonstrate
# coherence.

# pick an elliptic lcs and create more refined curve by interpolating
# we pick index 2 because it is the first not on a boundary of the domain,
# any could be choosen
c0 = elcs[2][1]
rcv0 = elcs[2][0]
rcvi = interp_curve(rcv0, 2500, per=1)

# create a filled circle which has a center a distance delta away from
# vortex center
delta = 5e-1
circ_delta = gen_filled_circ(1.5, 10000, c=c0 + delta)

# advect both the elliptic lcs and the nearby filled circle
frames = 401
adv_rcv, teval = flowmap_n(funcptr, t0, T, rcvi, params, n=frames)
adv_circ, _ = flowmap_n(funcptr, t0, T, circ_delta, params, n=frames)


# %%
# Animate
# -------
# Create animation of advected elliptic LCS and nearby filled circle.

# find which points from the filled circle are inside elliptic lcs
# for plotting purposes
mask = pts_in_poly_mask(rcv0, circ_delta)

# create plot
fig, ax = plt.subplots(dpi=100)
scatter0 = ax.scatter(adv_rcv[:, 0, 0] % period_x, adv_rcv[:, 0, 1], 1)
scatter_in = ax.scatter(
    adv_circ[mask, 0, 0] % period_x, adv_circ[mask, 0, 1], 0.5, "purple", zorder=0
)
scatter_out = ax.scatter(
    adv_circ[~mask, 0, 0] % period_x, adv_circ[~mask, 0, 1], 0.5, "orange", zorder=0
)
ax.set_xlim([domain[0][0] + 2, domain[0][1] - 2])
ax.set_ylim([domain[1][0], domain[1][1]])
ax.set_aspect("equal")
ax.set_title(f"t = {round(teval[0], 1)} days")


# function for animation
def update(frame):
    # for each frame, update the data stored on each artist
    x0 = adv_rcv[:, frame, 0] % period_x
    y0 = adv_rcv[:, frame, 1]
    x_in = adv_circ[mask, frame, 0] % period_x
    y_in = adv_circ[mask, frame, 1]
    x_out = adv_circ[~mask, frame, 0] % period_x
    y_out = adv_circ[~mask, frame, 1]
    data0 = np.column_stack((x0, y0))
    data_in = np.column_stack((x_in, y_in))
    data_out = np.column_stack((x_out, y_out))

    # update each scatter plot
    scatter0.set_offsets(data0)
    scatter_in.set_offsets(data_in)
    scatter_out.set_offsets(data_out)

    # update title
    ax.set_title(f"t = {round(teval[frame], 1)} days")

    return (scatter0, scatter_in, scatter_out)


# create animation
ani = FuncAnimation(fig=fig, func=update, frames=frames, interval=50)
plt.show()

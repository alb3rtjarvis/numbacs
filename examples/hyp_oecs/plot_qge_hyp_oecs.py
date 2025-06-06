#!/usr/bin/env python3
"""
Quasi-geostrophic hyperbolic OECS
=================================

Compute the hyperbolic OECS saddles for QGE flow.
"""

# Author: ajarvis
# Data: We thank Changhong Mou and Traian Iliescu for providing us with this dataset
#       and allowing it to be used here.

import numpy as np
from numbacs.diagnostics import S_eig_2D_data
from numbacs.extraction import hyperbolic_oecs
import matplotlib.pyplot as plt

# %%
# Get flow data
# --------------
# Load velocity data and set up domain.

# load in qge velocity data
u = np.load("../data/qge/qge_u.npy")
v = np.load("../data/qge/qge_v.npy")

# set up domain
nt, nx, ny = u.shape
x = np.linspace(0, 1, nx)
y = np.linspace(0, 2, ny)
t = np.linspace(0, 1, nt)
dx = x[1] - x[0]
dy = y[1] - y[0]
# %%
# S eigenvalues, eigenvectors
# ---------------------------
# Compute eigenvalues/vectors of S tensor from velocity field at time t = t[k0].

k0 = 15
# compute eigenvalues/vectors of Eulerian rate of strain tensor
eigvals, eigvecs = S_eig_2D_data(u[k0, :, :], v[k0, :, :], dx, dy)
s2 = eigvals[:, :, 1]

# %%
# Hyperbolic OECS saddles
# -----------------------
# Compute generalized saddle points and hyperbolic oecs.

# set parameters for hyperbolic_oecs function
r = 0.2
h = 1e-4
steps = 4000
maxlen = 0.05
minval = np.percentile(s2, 50)
n = 10

# compute hyperbolic_oecs
oecs = hyperbolic_oecs(s2, eigvecs, x, y, r, h, steps, maxlen, minval, n=n)

# %%
# Plot all OECS
# -------------
# Plot the OECS overlaid on iLE.
fig, ax = plt.subplots(dpi=200)
ax.contourf(x, y, s2.T, levels=np.linspace(0, np.percentile(s2, 99.5), 51), extend="both", zorder=0)

for k in range(len(oecs)):
    ax.plot(oecs[k][0][:, 0], oecs[k][0][:, 1], "r", lw=1)
    ax.plot(oecs[k][1][:, 0], oecs[k][1][:, 1], "b", lw=1)
ax.set_aspect("equal")
plt.show()
# %%
# Advect OECS
# -----------
# Advect OECS and a circle centered at the generalized saddle point.

# import necessary functions
from numbacs.flows import get_interp_arrays_2D, get_flow_2D
from numbacs.utils import gen_filled_circ
from numbacs.integration import flowmap_n

# get funcptr, set parameters for integration, and integrate
grid_vel, C_eval_u, C_eval_v = get_interp_arrays_2D(t, x, y, u, v)
funcptr = get_flow_2D(grid_vel, C_eval_u, C_eval_v)

nc = 4000
nT = 4
T = 0.06
t_eval = np.linspace(0, T, nT)
adv_circ = []
adv_rep = []
adv_att = []
t0 = t[k0]
# advect the top 2 (in strength) OECS
for k in range(len(oecs[:3])):
    circ1 = gen_filled_circ(maxlen, nc, c=oecs[k][2], xlims=(0, 1), ylims=(0, 2))
    adv_circ.append(flowmap_n(funcptr, t0, T, circ1, np.array([1.0]), n=nT)[0])
    adv_rep.append(flowmap_n(funcptr, t0, T, oecs[k][0], np.array([1.0]), n=nT)[0])
    adv_att.append(flowmap_n(funcptr, t0, T, oecs[k][1], np.array([1.0]), n=nT)[0])

# %%
# Plot advected OECS
# ------------------
# Plot advected OECS at 0.00, 0.02, 0.04, and 0.06 units of time after t0.
fig, axs = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, dpi=200)
axs = axs.flat
nax = len(axs)
for i in range(nax):
    kt = i
    axs[i].set_title(f"t0 + {t_eval[i]:.2f}")
    for k in range(len(adv_rep)):
        axs[i].scatter(
            adv_rep[k][:, kt, 0],
            adv_rep[k][:, kt, 1],
            1,
            "r",
            marker=".",
            edgecolors=None,
            linewidths=0,
        )
        axs[i].scatter(
            adv_att[k][:, kt, 0],
            adv_att[k][:, kt, 1],
            1,
            "b",
            marker=".",
            edgecolors=None,
            linewidths=0,
        )
        axs[i].scatter(adv_circ[k][:, kt, 0], adv_circ[k][:, kt, 1], 0.5, "g", zorder=0)
    axs[i].set_xlim([0, 1])
    axs[i].set_ylim([0, 2])
    axs[i].set_aspect("equal")
plt.show()

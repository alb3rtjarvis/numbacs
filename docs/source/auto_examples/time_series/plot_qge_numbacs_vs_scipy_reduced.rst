
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_examples/time_series/plot_qge_numbacs_vs_scipy_reduced.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_examples_time_series_plot_qge_numbacs_vs_scipy_reduced.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_time_series_plot_qge_numbacs_vs_scipy_reduced.py:


NumbaCS vs SciPy/NumPy -- QGE
=============================

Compare run times for flow map and FTLE between NumbaCS and
a pure SciPy/NumPy implementation for the QGE.

.. GENERATED FROM PYTHON SOURCE LINES 9-24

.. code-block:: Python


    # Author: ajarvis
    # Hardware: Intel(R) Core(TM) i7-3770K CPU @ 3.50GHz, Cores = 4, Threads = 8

    import numpy as np
    from scipy.interpolate import RegularGridInterpolator
    from scipy.integrate import odeint
    from numbacs.integration import flowmap_grid_2D
    from numbacs.flows import get_interp_arrays_2D, get_flow_2D
    from numbacs.diagnostics import ftle_grid_2D
    import time
    from math import copysign, log
    from multiprocessing import Pool
    from functools import partial








.. GENERATED FROM PYTHON SOURCE LINES 25-29

Get flow data and create NumbaCS flow
-------------------------------------
Load velocity data, set up domain, set the integration span and direction, create
interpolant of velocity data and retrieve necessary arrays.

.. GENERATED FROM PYTHON SOURCE LINES 29-65

.. code-block:: Python


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

    # use reduced domain or scipy will take much too long
    s = 4
    x = x[::s]
    y = y[::s]
    t = t[::s]
    u = u[::s, ::s, ::s]
    v = v[::s, ::s, ::s]

    nx = len(x)
    ny = len(y)

    # set integration span and integration direction
    t0 = 0.0
    T = 0.1
    params = np.array([copysign(1, T)])  # important this is an array of type float

    # get interpolant arrays of velocity field
    grid_vel, C_eval_u, C_eval_v = get_interp_arrays_2D(t, x, y, u, v)

    # get flow to be integrated
    funcptr = get_flow_2D(grid_vel, C_eval_u, C_eval_v, extrap_mode="linear")









.. GENERATED FROM PYTHON SOURCE LINES 66-69

Scipy interpolant and ODE function
----------------------------------
Create interpolant and function for SciPy ode solver.

.. GENERATED FROM PYTHON SOURCE LINES 69-81

.. code-block:: Python



    ui = RegularGridInterpolator((t, x, y), u, method="linear", bounds_error=False, fill_value=0.0)
    vi = RegularGridInterpolator((t, x, y), v, method="linear", bounds_error=False, fill_value=0.0)


    def odeint_fun(yy, tt):
        pt = np.array([tt, yy[0], yy[1]])

        return ui(pt)[0], vi(pt)[0]









.. GENERATED FROM PYTHON SOURCE LINES 82-88

SciPy flow map and FTLE functions
---------------------------------
Create functions to compute flow maps and FTLE using standard SciPy/Numpy methods.
Uses scipy.integrate.odeint (implements LSODA method) for particle integration.
The scipy.integrate.solve_ivp function is newer and allows the use of other solvers
but odeint is faster even when solve_ivp uses LSODA as its method.

.. GENERATED FROM PYTHON SOURCE LINES 88-120

.. code-block:: Python


    tspan = np.array([t0, t0 + T])


    def scipy_odeint_flowmap_par(t0, y0):
        tspan = np.array([t0, t0 + T])
        sol = odeint(odeint_fun, y0, tspan, rtol=1e-6, atol=1e-8)
        flowmap = sol[-1, :]

        return flowmap


    def numpy_ftle_par(fm, inds):
        i, j = inds
        absT = abs(T)
        dxdx = (fm[i + 1, j, 0] - fm[i - 1, j, 0]) / (2 * dx)
        dxdy = (fm[i, j + 1, 0] - fm[i, j - 1, 0]) / (2 * dy)
        dydx = (fm[i + 1, j, 1] - fm[i - 1, j, 1]) / (2 * dx)
        dydy = (fm[i, j + 1, 1] - fm[i, j - 1, 1]) / (2 * dy)

        off_diagonal = dxdx * dxdy + dydx * dydy
        C = np.array([[dxdx**2 + dydx**2, off_diagonal], [off_diagonal, dxdy**2 + dydy**2]])

        max_eig = np.linalg.eigvalsh(C)[-1]
        if max_eig > 1:
            ftle = 1 / (2 * absT) * log(max_eig)
        else:
            ftle = 0

        return ftle









.. GENERATED FROM PYTHON SOURCE LINES 121-126

Compute SciPy/Numpy flow map, FTLE
----------------------------------
Compute flowmap, FTLE, and calculate run times for the SciPy/NumPy implementation.
For this problem on this hardware, computing flow map and FTLE parallel in space
(as opposed to parallel in time) was the faster implementation.

.. GENERATED FROM PYTHON SOURCE LINES 126-172

.. code-block:: Python


    # set initial conditions
    n = 2
    t0span = np.linspace(0, 0.1, n)
    [X, Y] = np.meshgrid(x, y, indexing="ij")
    Y0 = np.column_stack((X.ravel(), Y.ravel()))
    sftle = np.zeros((n, nx - 2, ny - 2), np.float64)

    # set parallel pool to use maximum number of threads for this hardware,
    # open pool
    num_threads = 8
    pl = Pool(num_threads)

    # create inds to pass to ftle function
    xinds = np.arange(1, nx - 1)
    yinds = np.arange(1, ny - 1)
    [I, J] = np.meshgrid(xinds, yinds, indexing="ij")
    inds = np.column_stack((I.ravel(), J.ravel()))

    # compute flowmap and ftle parallel in space
    sfmtt = 0
    sftt = 0

    for k, t0 in enumerate(t0span):
        ks = time.perf_counter()
        func = partial(scipy_odeint_flowmap_par, t0)
        res = np.array(pl.map(func, Y0)).reshape(nx, ny, 2)
        kf = time.perf_counter()
        sfmtt += kf - ks

        fks = time.perf_counter()
        func2 = partial(numpy_ftle_par, res)
        sftle[k, :, :] = np.array(pl.map(func2, inds)).reshape(nx - 2, ny - 2)
        fkf = time.perf_counter()
        sftt += fkf - fks

    pl.close()
    pl.terminate()

    print("SciPy/NumPy flowmap and FTLE took " + f"{sfmtt + sftt:.5f} seconds for {n} iterates")
    print("Mean time for SciPy/NumPy flowmap and FTLE -- " + f"{(sfmtt + sftt) / n:.5f} seconds\n")
    print(f"Scipy flowmap took {sfmtt:.5} seconds for {n:1d} iterates")
    print(f"Mean time for Scipy flowmap -- {sfmtt / n:.5} seconds\n")
    print(f"NumPy ftle took {sftt:.5} seconds for {n:1d} iterates")
    print(f"Mean time for NumPy ftle -- {sftt / n:.5} seconds\n")





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    SciPy/NumPy flowmap and FTLE took 820.20150 seconds for 2 iterates
    Mean time for SciPy/NumPy flowmap and FTLE -- 410.10075 seconds

    Scipy flowmap took 820.06 seconds for 2 iterates
    Mean time for Scipy flowmap -- 410.03 seconds

    NumPy ftle took 0.14228 seconds for 2 iterates
    Mean time for NumPy ftle -- 0.071141 seconds





.. GENERATED FROM PYTHON SOURCE LINES 173-178

Compute NumbaCS flow map, FTLE
--------------------------------
Compute flowmap, FTLE, and calculate run times for the NumbaCS implementation.
For this problem on this hardware, computing flow map and FTLE parallel in space
(as opposed to parallel in time) was the faster implementation.

.. GENERATED FROM PYTHON SOURCE LINES 178-223

.. code-block:: Python


    ftle = np.zeros((n, nx, ny), np.float64)

    # first call and record warmup times
    wfm = time.perf_counter()
    flowmap_wu = flowmap_grid_2D(funcptr, t0, T, x, y, params)
    wu_fm = time.perf_counter() - wfm
    print(f"Flowmap with warm-up took {wu_fm:.5f} seconds")

    wf = time.perf_counter()
    ftle[0, :, :] = ftle_grid_2D(flowmap_wu, T, dx, dy)
    wu_f = time.perf_counter() - wf
    print(f"FTLE with warm-up took {wu_f:.5f} seconds")

    # initialize runtime counters
    fmtt = wu_fm
    ftt = wu_f

    # loop over initial times, compute flowmap and ftle
    for k, t0 in enumerate(t0span[1:]):
        ks = time.perf_counter()
        flowmap = flowmap_grid_2D(funcptr, t0, T, x, y, params)
        kf = time.perf_counter()
        fmtt += kf - ks

        fks = time.perf_counter()
        ftle[k, :, :] = ftle_grid_2D(flowmap, T, dx, dy)
        fkf = time.perf_counter()
        ftt += fkf - fks

    print("NumbaCS flowmap and FTLE took " + f"{fmtt + ftt:.5f} for {n:1d} iterates")
    print(f"Mean time for flowmap and FTLE -- {(fmtt + ftt) / n:.5f} seconds (w/ warmup)")
    print(
        "Mean time for flowmap and FTLE -- "
        + f"{(fmtt - wu_fm + ftt - wu_f) / (n - 1):.5f} seconds (w/o warmup)\n"
    )
    print(f"NumbaCS flowmap_grid_2D took {fmtt:.5f} seconds for {n:1d} iterates")
    print(f"Mean time for flowmap_grid_2D -- {fmtt / n:.5f} seconds (w/ warmup)")
    print(
        "Mean time for flowmap_grid_2D -- " + f"{(fmtt - wu_fm) / (n - 1):.5f} seconds (w/o warmup)\n"
    )
    print(f"NumbaCS ftle_grid_2D took {ftt:.5f} seconds for {n:1d} iterates")
    print(f"Mean time for ftle_grid_2D -- {ftt / n:.5f} seconds (w/ warmup)")
    print(f"Mean time for ftle_grid_2D -- {(ftt - wu_f) / (n - 1):.5f} seconds (w/o warmup)")





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    Flowmap with warm-up took 1.69785 seconds
    FTLE with warm-up took 0.00160 seconds
    NumbaCS flowmap and FTLE took 1.83230 for 2 iterates
    Mean time for flowmap and FTLE -- 0.91615 seconds (w/ warmup)
    Mean time for flowmap and FTLE -- 0.13285 seconds (w/o warmup)

    NumbaCS flowmap_grid_2D took 1.82910 seconds for 2 iterates
    Mean time for flowmap_grid_2D -- 0.91455 seconds (w/ warmup)
    Mean time for flowmap_grid_2D -- 0.13125 seconds (w/o warmup)

    NumbaCS ftle_grid_2D took 0.00320 seconds for 2 iterates
    Mean time for ftle_grid_2D -- 0.00160 seconds (w/ warmup)
    Mean time for ftle_grid_2D -- 0.00159 seconds (w/o warmup)




.. GENERATED FROM PYTHON SOURCE LINES 224-231

Compare timings
---------------
Compare timings and quantify speed-up. The second and third columns quantify the
speed-up gained using NumbaCS. The second column includes warm-up time, the speed-up
would increase as *n* grows larger. The third column ignores the warm-up time
and quantifies the speed-up as *n* goes to infinity and the warm-up time becomes
negligible. This represents the theoretical speed-up.

.. GENERATED FROM PYTHON SOURCE LINES 231-256

.. code-block:: Python



    stt = sfmtt + sftt
    ntt = fmtt + ftt

    stpi = (sfmtt + sftt) / n
    ntpi = (ntt - wu_fm - wu_f) / (n - 1)

    d1 = 5
    d2 = 2
    data = [
        [round(stt, d1), "--", "--"],
        [round(ntt, d1), round(stt / ntt, d2), round(stpi / ntpi, d2)],
    ]

    times = [f"total time (n={n})", "speedup", "speedup (n->inf)"]
    methods = ["SciPy/NumPy", "NumbaCS"]

    format_row = "{:>25}" * (len(data[0]) + 1)

    print(format_row.format("", *times))

    for name, vals in zip(methods, data):
        print(format_row.format(name, *vals))





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

                                      total time (n=2)                  speedup         speedup (n->inf)
                  SciPy/NumPy                 820.2015                       --                       --
                      NumbaCS                   1.8323                   447.63                  3087.04




.. GENERATED FROM PYTHON SOURCE LINES 257-266

.. note::

   The SciPy interpolation package creates a bottleneck when used to solve odes and has
   a large effect on the overall runtime. For this reason, we only run for 2
   iterates or the code would take much too long. As *n* increaes, the speed-up
   would increase quite quickly as the warm-up time of the NumbaCS implementation
   becomes less significant. Regardless, the NumbaCS implementation achieves a
   drastic speed-up when used on numerical velocity data. This is largely achieved
   by the numbalsoda and interpolation packages, both of which utilize Numba.


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (13 minutes 42.299 seconds)


.. _sphx_glr_download_auto_examples_time_series_plot_qge_numbacs_vs_scipy_reduced.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_qge_numbacs_vs_scipy_reduced.ipynb <plot_qge_numbacs_vs_scipy_reduced.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_qge_numbacs_vs_scipy_reduced.py <plot_qge_numbacs_vs_scipy_reduced.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_qge_numbacs_vs_scipy_reduced.zip <plot_qge_numbacs_vs_scipy_reduced.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_

:orphan:

Examples Gallery
================

This gallery contains all the NumbaCS examples.

.. note::
   Most of these examples are simply meant to demonstrate how to use the various
   modules in NumbaCS. Since many of the functions in NumbaCS are written with
   the Numba ``@njit`` decorator, they are optimized and compiled just-in-time
   and therefore, their first function call can be slow (especially for
   functions which call many compiled functions internally like
   :func:`numbacs.extraction.ftle_ordered_ridges`).
   For many examples here, these warmup times are included in the timings. The times
   listed at the end of each example should not be seen as representative of
   potential speed of NumbaCS when a time series of one of the diagnostics or
   extraction methods are desired. For more representative timings, see examples
   in the :ref:`auto_examples/index:time series` section.

.. note::
   To run any of these examples, matplotlib will need to be installed as it does
   not ship with NumbaCS. Also, to run any example which uses a numerical flow,
   the data will need to be downloaded from the
   `Github page <https://github.com/alb3rtjarvis/numbacs/tree/main/examples/data>`_.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>


Elliptic LCS
^^^^^^^^^^^^

This gallery contains examples for LAVD-based elliptic LCS.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compute the LAVD-based elliptic lcs for the bickley jet.">

.. only:: html

  .. image:: /auto_examples/elliptic_lcs/images/thumb/sphx_glr_plot_bickley_elliptic_lcs_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_elliptic_lcs_plot_bickley_elliptic_lcs.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Bickley jet Elliptic LCS</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compute the LAVD-based elliptic lcs for the QGE.">

.. only:: html

  .. image:: /auto_examples/elliptic_lcs/images/thumb/sphx_glr_plot_qge_elliptic_lcs_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_elliptic_lcs_plot_qge_elliptic_lcs.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Quasi-geostrophic Elliptic LCS</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/elliptic_lcs/plot_bickley_elliptic_lcs
   /auto_examples/elliptic_lcs/plot_qge_elliptic_lcs


Elliptic OECS
^^^^^^^^^^^^^

This gallery contains examples for IVD-based elliptic OECS.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compute the IVD-based elliptic OECS for the Bickley jet.">

.. only:: html

  .. image:: /auto_examples/elliptic_oecs/images/thumb/sphx_glr_plot_bickley_elliptic_oecs_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_elliptic_oecs_plot_bickley_elliptic_oecs.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Bickley jet Elliptic OECS</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compute the IVD-based elliptic OECS for the QGE.">

.. only:: html

  .. image:: /auto_examples/elliptic_oecs/images/thumb/sphx_glr_plot_qge_elliptic_oecs_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_elliptic_oecs_plot_qge_elliptic_oecs.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Quasi-geostrophic Elliptic OECS</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/elliptic_oecs/plot_bickley_elliptic_oecs
   /auto_examples/elliptic_oecs/plot_qge_elliptic_oecs

FTLE
^^^^

This gallery contains examples for FTLE fields and FTLE ridges.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compute the FTLE field for the bickley jet.">

.. only:: html

  .. image:: /auto_examples/ftle/images/thumb/sphx_glr_plot_bickley_ftle_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_ftle_plot_bickley_ftle.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Bickley jet FTLE</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compute the FTLE field and ridges for the bickley jet.">

.. only:: html

  .. image:: /auto_examples/ftle/images/thumb/sphx_glr_plot_bickley_ftle_ridges_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_ftle_plot_bickley_ftle_ridges.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Bickley jet FTLE ridges</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compute the FTLE field over the entire globe for ocean flow at time of using Copernicus reanalysis data.">

.. only:: html

  .. image:: /auto_examples/ftle/images/thumb/sphx_glr_plot_copernicus_globe_ftle_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_ftle_plot_copernicus_globe_ftle.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Copernicus Globe FTLE</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compute the FTLE field for the double gyre.">

.. only:: html

  .. image:: /auto_examples/ftle/images/thumb/sphx_glr_plot_dg_ftle_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_ftle_plot_dg_ftle.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Double gyre FTLE</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compute the FTLE field and ridges for the double gyre.">

.. only:: html

  .. image:: /auto_examples/ftle/images/thumb/sphx_glr_plot_dg_ftle_ridges_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_ftle_plot_dg_ftle_ridges.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Double gyre FTLE ridges</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compute the FTLE field for atmospheric flow at time of Godzilla dust storm using MERRA-2 data which is vertically averaged over pressure surfaces ranging from 500hPa to 800hPa.">

.. only:: html

  .. image:: /auto_examples/ftle/images/thumb/sphx_glr_plot_merra_ftle_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_ftle_plot_merra_ftle.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MERRA-2 FTLE</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compute the FTLE field and ridges for atmospheric flow at time of Godzilla dust storm using MERRA-2 data which is vertically averaged over pressure surfaces ranging from 500hPa to 800hPa.">

.. only:: html

  .. image:: /auto_examples/ftle/images/thumb/sphx_glr_plot_merra_ftle_ridges_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_ftle_plot_merra_ftle_ridges.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MERRA-2 FTLE ridges</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compute the FTLE field over the entire globe for atmospheric flow at time of Godzilla dust storm using MERRA-2 data which is vertically averaged over pressure surfaces ranging from 500hPa to 800hPa.">

.. only:: html

  .. image:: /auto_examples/ftle/images/thumb/sphx_glr_plot_merra_globe_ftle_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_ftle_plot_merra_globe_ftle.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MERRA-2 Globe FTLE</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compute the FTLE field for the QGE.">

.. only:: html

  .. image:: /auto_examples/ftle/images/thumb/sphx_glr_plot_qge_ftle_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_ftle_plot_qge_ftle.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Quasi-geostrophic FTLE</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compute the FTLE field and ridges for the QGE.">

.. only:: html

  .. image:: /auto_examples/ftle/images/thumb/sphx_glr_plot_qge_ftle_ridges_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_ftle_plot_qge_ftle_ridges.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Quasi-geostrophic FTLE ridges</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/ftle/plot_bickley_ftle
   /auto_examples/ftle/plot_bickley_ftle_ridges
   /auto_examples/ftle/plot_copernicus_globe_ftle
   /auto_examples/ftle/plot_dg_ftle
   /auto_examples/ftle/plot_dg_ftle_ridges
   /auto_examples/ftle/plot_merra_ftle
   /auto_examples/ftle/plot_merra_ftle_ridges
   /auto_examples/ftle/plot_merra_globe_ftle
   /auto_examples/ftle/plot_qge_ftle
   /auto_examples/ftle/plot_qge_ftle_ridges

Hyperbolic LCS
^^^^^^^^^^^^^^^

This gallery contains examples hyperbolic LCS found using the variational method.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compute hyperbolic LCS using the variational theory for the double gyre.">

.. only:: html

  .. image:: /auto_examples/hyp_lcs/images/thumb/sphx_glr_plot_dg_hyp_lcs_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_hyp_lcs_plot_dg_hyp_lcs.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Double gyre Hyperbolic LCS</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compute hyperbolic LCS using the variational theory for atmospheric flow at time of Godzilla dust storm using MERRA-2 data which is vertically averaged over pressure surfaces ranging from 500hPa to 800hPa..">

.. only:: html

  .. image:: /auto_examples/hyp_lcs/images/thumb/sphx_glr_plot_merra_hyp_lcs_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_hyp_lcs_plot_merra_hyp_lcs.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MERRA-2 Hyperbolic LCS</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/hyp_lcs/plot_dg_hyp_lcs
   /auto_examples/hyp_lcs/plot_merra_hyp_lcs

Hyperbolic OECS
^^^^^^^^^^^^^^^

This gallery contains examples for iLE fields and hyperbolic OECS.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compute the iLE field for the bickley jet.">

.. only:: html

  .. image:: /auto_examples/hyp_oecs/images/thumb/sphx_glr_plot_bickley_ile_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_hyp_oecs_plot_bickley_ile.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Bickley jet iLE</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compute the iLE field for the double gyre.">

.. only:: html

  .. image:: /auto_examples/hyp_oecs/images/thumb/sphx_glr_plot_dg_ile_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_hyp_oecs_plot_dg_ile.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Double gyre iLE</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compute the hyperbolic OECS saddles for atmospheric flow at time of Godzilla dust storm using MERRA-2 data which is vertically averaged over pressure surfaces ranging from 500hPa to 800hPa.">

.. only:: html

  .. image:: /auto_examples/hyp_oecs/images/thumb/sphx_glr_plot_merra_hyp_oecs_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_hyp_oecs_plot_merra_hyp_oecs.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MERRA-2 hyperbolic OECS</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compute the iLE field and for atmospheric flow at time of Godzilla dust storm using MERRA-2 data which is vertically averaged over pressure surfaces ranging from 500hPa to 800hPa.">

.. only:: html

  .. image:: /auto_examples/hyp_oecs/images/thumb/sphx_glr_plot_merra_ile_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_hyp_oecs_plot_merra_ile.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MERRA iLE</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compute the hyperbolic OECS saddles for QGE flow.">

.. only:: html

  .. image:: /auto_examples/hyp_oecs/images/thumb/sphx_glr_plot_qge_hyp_oecs_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_hyp_oecs_plot_qge_hyp_oecs.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Quasi-geostrophic hyperbolic OECS</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compute the iLE field for the QGE.">

.. only:: html

  .. image:: /auto_examples/hyp_oecs/images/thumb/sphx_glr_plot_qge_ile_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_hyp_oecs_plot_qge_ile.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Quasi-geostraphic iLE</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/hyp_oecs/plot_bickley_ile
   /auto_examples/hyp_oecs/plot_dg_ile
   /auto_examples/hyp_oecs/plot_merra_hyp_oecs
   /auto_examples/hyp_oecs/plot_merra_ile
   /auto_examples/hyp_oecs/plot_qge_hyp_oecs
   /auto_examples/hyp_oecs/plot_qge_ile

Time series
^^^^^^^^^^^

This gallery contains examples for time series of diagnostics and
flow map composition.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compare run times for flow map and FTLE between NumbaCS and a pure SciPy/NumPy implementation for the double gyre.">

.. only:: html

  .. image:: /auto_examples/time_series/images/thumb/sphx_glr_plot_dg_numbacs_vs_scipy_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_time_series_plot_dg_numbacs_vs_scipy.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">NumbaCS vs SciPy/NumPy -- Double gyre</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compare run times for different flowmap methods for the double gyre.">

.. only:: html

  .. image:: /auto_examples/time_series/images/thumb/sphx_glr_plot_dg_time_series_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_time_series_plot_dg_time_series.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Double gyre time series</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compare run times for flow map and FTLE between NumbaCS and a pure SciPy/NumPy implementation for the QGE.">

.. only:: html

  .. image:: /auto_examples/time_series/images/thumb/sphx_glr_plot_qge_numbacs_vs_scipy_reduced_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_time_series_plot_qge_numbacs_vs_scipy_reduced.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">NumbaCS vs SciPy/NumPy -- QGE</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compute a time series of FTLE fields and ridges for the QGE.">

.. only:: html

  .. image:: /auto_examples/time_series/images/thumb/sphx_glr_plot_qge_ridge_time_series_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_time_series_plot_qge_ridge_time_series.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Quasi-geostrophic FTLE ridges time series</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compare run times for different flowmap methods for the QGE.">

.. only:: html

  .. image:: /auto_examples/time_series/images/thumb/sphx_glr_plot_qge_time_series_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_time_series_plot_qge_time_series.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Quasi-geostrophic time series</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/time_series/plot_dg_numbacs_vs_scipy
   /auto_examples/time_series/plot_dg_time_series
   /auto_examples/time_series/plot_qge_numbacs_vs_scipy_reduced
   /auto_examples/time_series/plot_qge_ridge_time_series
   /auto_examples/time_series/plot_qge_time_series


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: auto_examples_python.zip </auto_examples/auto_examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: auto_examples_jupyter.zip </auto_examples/auto_examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_

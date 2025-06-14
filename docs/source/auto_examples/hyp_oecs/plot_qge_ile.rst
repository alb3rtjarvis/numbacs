
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_examples/hyp_oecs/plot_qge_ile.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_examples_hyp_oecs_plot_qge_ile.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_hyp_oecs_plot_qge_ile.py:


Quasi-geostraphic iLE
=====================

Compute the iLE field for the QGE.

.. GENERATED FROM PYTHON SOURCE LINES 8-16

.. code-block:: Python


    # Author: ajarvis
    # Data: We thank Changhong Mou and Traian Iliescu for providing us with this dataset
    #       and allowing it to be used here.

    import numpy as np
    import matplotlib.pyplot as plt
    from numbacs.diagnostics import ile_2D_data







.. GENERATED FROM PYTHON SOURCE LINES 17-20

Get flow data
--------------
Load velocity data and set up domain.

.. GENERATED FROM PYTHON SOURCE LINES 20-34

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









.. GENERATED FROM PYTHON SOURCE LINES 35-38

iLE
----
Compute iLE field from velocity data directly at time t[k].

.. GENERATED FROM PYTHON SOURCE LINES 38-41

.. code-block:: Python

    k = 15
    ile = ile_2D_data(u[k, :, :], v[k, :, :], dx, dy)








.. GENERATED FROM PYTHON SOURCE LINES 42-45

Plot
----
Plot the results.

.. GENERATED FROM PYTHON SOURCE LINES 45-51

.. code-block:: Python

    fig, ax = plt.subplots(dpi=200)
    ax.contourf(
        x, y, ile.T, levels=np.linspace(0, np.percentile(ile, 99.5), 51), extend="both", zorder=0
    )
    ax.set_aspect("equal")
    plt.show()



.. image-sg:: /auto_examples/hyp_oecs/images/sphx_glr_plot_qge_ile_001.png
   :alt: plot qge ile
   :srcset: /auto_examples/hyp_oecs/images/sphx_glr_plot_qge_ile_001.png
   :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 1.606 seconds)


.. _sphx_glr_download_auto_examples_hyp_oecs_plot_qge_ile.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_qge_ile.ipynb <plot_qge_ile.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_qge_ile.py <plot_qge_ile.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_qge_ile.zip <plot_qge_ile.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_

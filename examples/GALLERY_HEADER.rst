Examples Gallery
================

This gallery contains all the NumbaCS examples.

.. note::
   Most of these examples are simply meant to demonstrate how to use the various
   modules in NumbaCS. Since many of the functions in NumbaCS are written with
   the Numba ``@njit`` decorator, they are optimized and compiled just-in-time
   and therefore, their first function call can be slow (especially for
   complicated functions like :func:`numbacs.extraction.ftle_ordered_ridges`).
   For each example here, these warmup times are included in the timings. The times
   listed at the end of each example should not be seen as representative of
   potential speed of NumbaCS when a time series of one of the diagnostics or
   extraction methods are desired. For more representative timing, see examples
   in the :ref:`auto_examples/index:time series` section.
   
   To run any of these examples, matplotlib will need to be installed as it does
   not ship with NumbaCS. Also, to run any example which uses a numerical flow,
   the data will need to be downloaded from the 
   `Github page <https://github.com/alb3rtjarvis/numbacs>`_.

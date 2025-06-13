Contributing to NumbaCS
=======================


First off, thank you for considering contributing to NumbaCS. 

How to contribute
-----------------

The preferred workflow for contributing to NumbaCS is to fork the
[main repository](https://github.com/alb3rtjarvis/NumbaCS) on
GitHub, clone, and develop on a branch. Steps:

1. Fork the [project repository](https://github.com/alb3rtjarvis/NumbaCS)
   by clicking on the 'Fork' button near the top right of the page. This creates
   a copy of the code under your GitHub user account. For more details on
   how to fork a repository see [this guide](https://help.github.com/articles/fork-a-repo/).

2. Clone your fork of the NumbaCS repo from your GitHub account to your local disk:

   ```bash
   $ git clone git@github.com:YourLogin/numbacs.git
   $ cd numbacs
   ```

3. Create a ``feature`` branch to hold your development changes:

   ```bash
   $ git checkout -b my-feature
   ```

   Always use a ``feature`` branch. It's good practice to never work on the ``main`` branch!

4. Develop the feature on your feature branch. Add changed files using ``git add`` and then ``git commit`` files:

   ```bash
   $ git add modified_files
   $ git commit
   ```

   to record your changes in Git, then push the changes to your GitHub account with:

   ```bash
   $ git push -u origin my-feature
   ```

5. Follow [these instructions](https://help.github.com/articles/creating-a-pull-request-from-a-fork)
to create a pull request from your fork. This will send an email to the committers.

(If any of the above seems like magic to you, please look up the
[Git documentation](https://git-scm.com/documentation) on the web, or ask a friend or another contributor for help.)

Pull Request Checklist
----------------------

We recommended that your contribution complies with the
following rules before you submit a pull request:

-  If your pull request addresses an issue, please use the pull request title
   to describe the issue and mention the issue number in the pull request description. This will make sure a link back to the original issue is
   created.

-  All public functions/methods should have informative docstrings with sample
   usage presented as doctests when appropriate.

-  Please prefix the title of your pull request with `[MRG]` (Ready for
   Merge), if the contribution is complete and ready for a detailed review.
   A developer will review your code and change the prefix of the pull
   request to `[MRG + 1]` on approval, making it eligible
   for merging. An incomplete contribution -- where you expect to do more work before
   receiving a full review -- should be prefixed `[WIP]` (to indicate a work
   in progress) and changed to `[MRG]` when it matures. WIPs may be useful
   to: indicate you are working on something to avoid duplicated work,
   request broad review of functionality or API, or seek collaborators.
   WIPs often benefit from the inclusion of a
   [task list](https://github.com/blog/1375-task-lists-in-gfm-issues-pulls-comments)
   in the PR description.

-  When adding additional functionality, provide at least one
   example script in the ``examples/`` folder. Have a look at other
   examples for reference. Examples should demonstrate why the new
   functionality is useful in practice and, if possible, compare it
   to other methods available in NumbaCS.

-  Documentation and tests are necessary for enhancements to be
   accepted. Bug-fixes or new features should be provided with 
   with passing tests using pytest. If you are unfamiliar with pytest,
   see the [pytest](https://docs.pytest.org/en/stable/) documentation for
   details. These tests verify the correct behavior of the fix or feature. 
   For the Bug-fixes case, at the time of the PR, tests should fail for
   the code base in main and pass for the PR code.

-  At least one paragraph of narrative documentation with links to
   references in the literature (with PDF links when possible) and
   the example.

Bonus points for contributions that include a performance analysis with
a benchmark script and profiling output (please report on the GitHub issue).

Filing bugs
-----------
We use Github issues to track all bugs and feature requests; feel free to
open an issue if you have found a bug or wish to see a feature implemented.

It is recommended to check that your issue complies with the
following rules before submitting:

-  Verify that your issue is not being currently addressed by other
   [issues](https://github.com/alb3rtjarvis/numbacs/issues)
   or [pull requests](https://github.com/alb3rtjarvis/numbacs/pulls).

-  Please ensure all code snippets and error messages are formatted in
   appropriate code blocks.
   See [Creating and highlighting code blocks](https://help.github.com/articles/creating-and-highlighting-code-blocks).

-  Please include your operating system type and version number, as well
   as your Python, NumbaCS, Numba, NumPy, SciPy, numbalsoda, interpolation, and
   ContourPy versions. This information
   can be found by running the following code snippet:

  ```python
  import platform; print(platform.platform())
  import sys; print("Python", sys.version)
  import numbacs; print("NumbaCS", numbacs.__version__)
  import numba; print("Numba", numba.__version__)
  import numpy; print("NumPy", numpy.__version__)
  import scipy; print("SciPy", scipy.__version__)
  import contourpy; print("ContourPy", contourpy.__version__)
  ```
  
  The rest (numbalsoda and interpolation) can be found by opening a terminal and
  running the following commands:
  
  ```bash
  $ conda activate numbacs_env
  $ conda list
  ```
  
  where ```numbacs_env``` is the conda enviornment that NumbaCS is installed in.

-  Please be specific about what functions are involved
   and the shape of the data, as appropriate; please include a
   [reproducible](http://stackoverflow.com/help/mcve) code snippet
   or link to a [gist](https://gist.github.com). If an exception is raised,
   please provide the traceback.

Documentation
-------------

We are glad to accept any sort of documentation: function docstrings,
reStructuredText documents (like this one), tutorials, etc.
reStructuredText documents live in the source code repository under the
doc/ directory.

You can edit the documentation using any text editor and then generate
the HTML output by typing ``make html`` from the ``docs/`` directory.
Alternatively, ``make`` can be used to quickly generate the
documentation without the example gallery with `make html-noplot`. The resulting HTML files will
be placed in `docs/build/html/` and are viewable in a web browser.

For building the documentation, you will need
[sphinx](https://www.sphinx-doc.org/),
[matplotlib](http://matplotlib.org/), and
[pillow](http://pillow.readthedocs.io/en/latest/).

When you are writing documentation, it is important to keep a good
compromise between mathematical and algorithmic details, and give
intuition to the reader on what the algorithm does. It is best to always
start with a small paragraph with a hand-waving explanation of what the
method does to the data and a figure (coming from an example)
illustrating it.


This Contribution guide is strongly inspired by the one of the 
[POT](https://github.com/PythonOT/POT) team, which was inspired by the one
of the [scikit-learn](https://github.com/scikit-learn/scikit-learn) team.

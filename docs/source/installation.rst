Installation
############

The source code for xqubit is available on
`GitHub <https://github.com/bitdessin/xqubit>`_,
and the package can be installed from
`PyPI <https://pypi.org/project/xqubit/>`_.
The recommended installation method is via pip:

.. code-block:: bash

    pip install xqubit


For R users, xqubit can be accessed through the
`reticulate <https://rstudio.github.io/reticulate/>`_ package,
which provides an interface to Python.
To install xqubit from within R, run:

.. code-block:: r

    library(reticulate)

    py_install("xqubit", pip = TRUE)


This installs xqubit into the Python environment associated with reticulate.
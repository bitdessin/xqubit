API Reference
=============

Expression Data
---------------

The :class:`xqubit.ExpressionData` class stores a gene expression matrix
together with sample annotation, including experimental conditions, time
points, and biological replicates.

It provides a consistent interface for accessing expression data as a
gene × sample matrix, a gene × replicate × condition array, or a
gene × condition matrix after replicate averaging. This structure allows
downstream functions to handle replicate-aware data and condition-level
summaries consistently.

.. autoclass:: xqubit.ExpressionData
   :members:
   :undoc-members:
   :show-inheritance:


Data Utilities
--------------

The :mod:`xqubit.utils` module provides helper functions for reading and
standardizing gene expression data.

It accepts expression data from arrays, data frames, or tabular files, and
checks that expression values, gene identifiers, and sample annotation are
properly aligned before analysis.

.. automodule:: xqubit.utils
   :members:
   :undoc-members:



Similarity and Statistical Metrics
----------------------------------

The :mod:`xqubit.metrics` module provides functions for calculating gene-wise
scores and gene-gene similarity matrices.

It includes basic gene-wise statistics, Pearson correlation, cosine-squared
similarity, and fidelity between state vectors. These functions can be used
for gene filtering, similarity analysis, and network construction.

.. automodule:: xqubit.metrics
   :members:
   :undoc-members:


State-Vector Encoding
---------------------

The :mod:`xqubit.qstate` module provides functions for encoding temporal gene
expression profiles as normalized complex-valued state vectors.

Encoding methods define how expression magnitude and temporal changes are
mapped to amplitudes and phases. The resulting state vectors can be compared
by fidelity and used to construct gene networks.

.. automodule:: xqubit.qstate
   :members:
   :undoc-members:



SWAP Test Simulation
--------------------

The :mod:`xqubit.qcircuit` module provides circuit-based fidelity estimation
using the SWAP test.

This module is mainly intended for checking or demonstrating how fidelity can
be estimated from state vectors using simulated quantum measurements. For
large-scale gene-gene similarity analysis, direct numerical fidelity
calculation is usually more practical.

.. automodule:: xqubit.qcircuit
   :members:
   :undoc-members:


Network Construction and Community Analysis
-------------------------------------------

The :mod:`xqubit.nx` module provides tools for constructing fidelity-based
or other similarity-based gene networks and detecting gene communities.

It includes network construction by similarity thresholding and k-nearest
neighbor sparsification, Leiden-based community detection, parameter scanning,
parameter ranking, and visualization utilities. These functions support the
exploration of transcriptomic structure from gene-gene similarity matrices.

.. automodule:: xqubit.nx
   :members:
   :undoc-members:

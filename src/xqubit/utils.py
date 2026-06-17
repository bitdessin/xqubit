import logging
import os
import re
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig()
logger = logging.getLogger(__name__)


_COLORS = ["#557ca7", "#e98a2e", "#d5525a", "#80b9b2", "#66a14f",
           "#e9c74c", "#aa7aa0", "#f79ba8", "#98745e", "#b9b0ab"]

def _close_plt(file_name):
    fig = plt.gcf()
    fig.tight_layout()
    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()
    plt.close()


def _init_gene_names(n_genes: int) -> list[str]:
    if n_genes <= 0:
        raise ValueError("n_genes must be positive.")
    w = len(str(n_genes))
    return [f"gene_{i:0{w}d}" for i in range(1, n_genes + 1)]


def read_data(
    x: str | np.ndarray | pd.DataFrame,
    design: dict | pd.DataFrame | None = None,
    genes: Sequence[str] | None = None,
    **kwargs
) -> dict:
    """
    Read gene expression data and sample annotation.

    This function converts input expression data into a standard format used
    by this package: a numeric gene × sample expression matrix, a sample
    annotation table, and gene identifiers.

    Parameters
    ----------
    x : str, os.PathLike, numpy.ndarray, or pandas.DataFrame
        Gene expression data.

        - If a file path is given, the file is read with ``pandas.read_csv``.
        - If a ``numpy.ndarray`` is given, it must have shape
          (n_genes, n_samples), and ``design`` must also be provided.
        - If a ``pandas.DataFrame`` is given, rows are treated as genes and
          columns are treated as samples.

    design : dict or pandas.DataFrame, optional
        Sample annotation table with one row per sample.

        The table must contain the following columns:

        - ``condition``: experimental condition, time point, or stage
        - ``replicate``: biological replicate identifier

        The row order of ``design`` must match the column order of the
        expression matrix.

        If ``design`` is ``None``, sample annotation is inferred from the
        column names of ``x``. In this case, column names must follow the
        format ``<condition>_<replicate>``, for example ``72h_1``,
        ``control_2``, or ``DAG10_R1``.

    genes : sequence of str, optional
        Gene identifiers corresponding to the rows of the expression matrix.

        If ``genes`` is ``None`` and ``x`` is a ``pandas.DataFrame`` with a
        non-integer index, the DataFrame index is used as gene identifiers.
        Otherwise, gene identifiers are generated automatically.

    **kwargs
        Additional keyword arguments passed to ``pandas.read_csv`` when
        ``x`` is a file path.
        
        If the input file contains gene identifiers in the first column,
        pass ``index_col=0`` so that the first column is used as the row index
        rather than being treated as expression values.


    Returns
    -------
    dict
        Dictionary containing standardized expression data.

        - ``"exp"``: ``numpy.ndarray`` with shape (n_genes, n_samples)
          Expression matrix stored as ``float64``.
        - ``"design"``: ``pandas.DataFrame`` with shape (n_samples, n_fields)
          Sample annotation table containing at least ``condition``, ``timepoint``, and ``replicate``.
        - ``"genes"``: list of str
          Gene identifiers aligned with the rows of ``"exp"``. 
    """
    # expression data    
    if isinstance(x, np.ndarray):
        logger.debug("Load expression data from numpy array.")
        exp_mat = x.astype(np.float64)
        if (design is None) or (not isinstance(design, (dict, pd.DataFrame))):
            raise ValueError("Experimental design must be provided as dict or DataFrame when input is a numpy array.")
        if genes is None:
            genes = _init_gene_names(exp_mat.shape[0])
    
    else:
        logger.debug("Load expression data from pandas DataFrame or file.")
        if isinstance(x, (str, os.PathLike)):
            logger.debug(f"Load expression data from file: {x}")
            if not os.path.exists(x):
                raise FileNotFoundError(x)
            x = pd.read_csv(x, **kwargs)
        
        exp_mat = x.to_numpy().astype(np.float64)

        if genes is None:
            # if x.index.dtype == "int64":
            if pd.api.types.is_integer_dtype(x.index):
                genes = _init_gene_names(x.shape[0])
            else:
                genes = x.index.to_list()

        if design is None:
            design = {"sample": x.columns.to_list(), "condition": [], "timepoint": [], "replicate": []}
            for c in design["sample"]:
                try:
                    s, r = c.rsplit("_", 1)
                    design["condition"].append(s)
                    design["timepoint"].append(__find_timepoint(s))
                    design["replicate"].append(r)
                except ValueError:
                    raise ValueError(f"Column name '{c}' must follow '<condition>_<replicate>' format.")

    genes = list(genes)
    design = pd.DataFrame(design).reset_index(drop=True)

    logger.debug(f"Expression matrix: {exp_mat.shape=}")
    logger.debug(f"Experimental design: {design}")
    logger.info(f"Expression data ({len(genes)} genes x {design.shape[0]} samples) loaded successfully.")

    # validation
    if len(genes) != exp_mat.shape[0]:
        raise ValueError("Length of genes must match number of rows in expression matrix.")
    if len(design) != exp_mat.shape[1]:
        raise ValueError("Number of rows in design must match number of samples.")
    if not {"condition", "replicate"}.issubset(design.columns):
        raise ValueError("Design dictionary must contain 'condition' and 'replicate' keys.")
    if not np.isfinite(exp_mat).all():
        raise ValueError("Expression matrix contains NaN or infinite values.")

    return {"exp": exp_mat, "design": design, "genes": genes}


def __find_timepoint(s):
    ts = np.nan
    
    m = re.search(r"\d+", s)
    if m is not None:
        ts = int(m.group(0))

    return ts
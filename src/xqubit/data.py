import copy
import logging
from typing import Sequence

import numpy as np
import pandas as pd
import sklearn

logging.basicConfig()
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt

from . import utils


class ExpressionData:
    """
    Container for gene expression data with experimental design.

    This class stores a gene expression matrix together with experimental design
    information, such as conditions, time points, and biological replicates.

    The input matrix is expected to have genes in rows and samples in columns.
    Internally, the data are also represented as a three dimensional array with
    the shape gene × replicate × condition.
    """
    
    def __init__(
        self,
        x: np.ndarray | pd.DataFrame,
        design: dict | pd.DataFrame,
        genes: Sequence[str] | None = None
    ) -> None:
        """
        Create an ExpressionData object.

        Parameters
        ----------
        x : numpy.ndarray or pandas.DataFrame
            Gene expression matrix with shape (n_genes, n_samples).
            Rows correspond to genes and columns correspond to samples.
            Values should be normalized expression values, such as
            variance-stabilized counts, log-transformed counts, or another
            comparable expression scale.

        design : dict or pandas.DataFrame
            Sample annotation table with one row per sample.

            The table must contain the following columns:
            - ``condition``: experimental condition, time point, or stage
            - ``replicate``: biological replicate identifier
            - ``timepoint``: time point of the sample in integers (optional)

            The row order of ``design`` must match the column order of ``x``.

        genes : sequence of str, optional
            Gene identifiers corresponding to the rows of ``x``.
            If ``None``, gene identifiers are generated automatically.

        Notes
        -----
        Input data are standardized by ``utils.read_data`` before being stored.
        The expression matrix is then reshaped into an array with shape
        (n_genes, n_replicates, n_conditions). Conditions and replicates are
        ordered by their sorted unique values in the design table.
        """
        data = utils.read_data(x, design, genes)
        
        self.__expmat = data["exp"]
        self.__genes = data["genes"]
        self.__design = pd.DataFrame(data["design"])

        # expression
        self.__expcube, self.__designcube, self.__sample_map = self.__melt_exp(self.__expmat, self.__design)
        


    def __melt_exp(self, x, design) -> tuple[np.ndarray, pd.DataFrame, list[tuple[int, int]]]:
        
        """
        Convert a gene × sample expression matrix into a gene × replicate ×
        condition array.

        Parameters
        ----------
        x : numpy.ndarray, shape (n_genes, n_samples)
            Gene expression matrix.

        design : pandas.DataFrame, shape (n_samples, n_fields)
            Sample annotation table containing ``condition`` and ``replicate``
            columns.

        Returns
        -------
        exp_cube : numpy.ndarray, shape (n_genes, n_replicates, n_conditions)
            Expression values arranged by gene, replicate, and condition.

        design_cube : pandas.DataFrame
            Condition-level annotation aligned with the condition axis of
            ``exp_cube``. The ``replicate`` column is removed.

        sample_map : list of tuple[int, int]
            Mapping from each original sample column to its corresponding
            replicate and condition indices in ``exp_cube``.
        """
        if x.shape[1] != len(design):
            raise ValueError(f"Expression matrix columns ({x.shape[1]}) do not match design rows ({len(design)})")

        uniq_c = sorted(set(design["condition"]))
        uniq_r = sorted(set(design["replicate"]))
        c2i = {c: i for i, c in enumerate(uniq_c)}
        r2i = {r: i for i, r in enumerate(uniq_r)}
        design_cols = [col for col in design.columns if col != "replicate"]

        sample_map = []
        exp_cube = np.full((x.shape[0], len(uniq_r), len(uniq_c)), np.nan, dtype=np.float64)
        for col_idx, (s, r) in enumerate(zip(design["condition"], design["replicate"])):
            if (r2i[r], c2i[s]) in sample_map:
                raise ValueError("Duplicate combination of (condition, replicate) detected in design.")
            exp_cube[:, r2i[r], c2i[s]] = x[:, col_idx]
            sample_map.append((r2i[r], c2i[s]))

        design_rows = []
        for cond in uniq_c:
            d_cond = design.loc[design["condition"] == cond, design_cols]
            if d_cond.empty:
                raise ValueError(f"Condition '{cond}' not found in design table.")
            design_rows.append(d_cond.iloc[0].to_dict())

        design_cube = pd.DataFrame(
            design_rows,
            index=pd.Index(uniq_c, name="condition"),
            columns=design_cols,
        )

        #if np.isnan(exp_cube).any():
        #    raise ValueError("Missing values detected in expression cube. Check design completeness.")

        logger.debug(f"Expression tensor was built with shape: {exp_cube.shape}")
        return exp_cube, design_cube, sample_map
    

    def __repr__(self) -> str:
        x = "ExpressionData\n"

        cl = []
        for c_, r_ in zip(self.design("condition"), self.design("replicate")):
            cl.append(f"{c_}({r_})")
        exp_ = pd.DataFrame(self.__expmat, index=self.__genes, columns=cl)

        x += f'Gene Expression Data ({self.__expmat.shape[0]} x {self.__expmat.shape[1]}):\n'
        x += f"{exp_.head(5).to_string()}\n"
        if exp_.shape[1] > 5:
            x += "...\n"
        
        return x

    
    def genes(
        self, i: int | str | None = None
    ) -> list[str] | str | int:
        """
        Access gene identifiers.

        Parameters
        ----------
        i : int, str, or None, optional
            Gene selector.

            - If ``None``, return all gene identifiers.
            - If ``int``, return the gene identifier at that row index.
            - If ``str``, return the row index of the specified gene.

        Returns
        -------
        list of str, str, or int
            Gene identifiers or a row index, depending on ``i``.
        """
        g = None
        if i is None:
            g = self.__genes.copy()
        else:
            if isinstance(i, int):
                if i < 0 or i >= len(self.__genes):
                    raise IndexError(f"Gene index out of range: {i}")
                g = self.__genes[i]
            elif isinstance(i, str):
                if i not in self.__genes:
                    raise ValueError(f"Gene name not found: {i}")
                g = self.__genes.index(i)
            else:
                raise TypeError(f"Invalid type for gene index: {type(i)}. Must be int or str.")
        return g


    def design(self, att: str | None = None, flatten: bool = False) -> pd.DataFrame | pd.Series:
        """
        Access sample or condition annotation.

        Parameters
        ----------
        att : str or None, optional
            Annotation column to retrieve. If ``None``, return the full
            annotation table.

        flatten : bool, default=False
            If ``False``, return condition-level annotation aligned with the
            condition axis of the expression array.

            If ``True``, return sample-level annotation aligned with the
            original sample order.

        Returns
        -------
        pandas.DataFrame or pandas.Series
            Annotation table or one annotation column.
        """
        if flatten:
            d = copy.deepcopy(self.__design)
        else:
            d = copy.deepcopy(self.__designcube)
        
        if att is None:
            return d
        if att not in d.columns:
            raise ValueError(f"Attribute not found in design: {att}")
        
        return d[att].copy()
        


    
    def exp(
        self,
        avg_replicates: bool = False,
        zscore: bool = False,
        flatten: bool = False
    ) -> np.ndarray:
        """
        Retrieve expression values.

        Parameters
        ----------
        avg_replicates : bool, default=False
            If ``True``, average biological replicates for each condition and
            return a gene × condition matrix.

        zscore : bool, default=False
            If ``True``, standardize expression values for each gene across the
            last axis of the returned array.

        flatten : bool, default=False
            If ``True``, return a gene × sample matrix aligned with the original
            sample order. This option applies when replicates have not been
            averaged.

        Returns
        -------
        numpy.ndarray
            Expression values. Depending on the options, the returned array has
            one of the following shapes:

            - gene × replicate × condition
            - gene × condition
            - gene × sample
        """
        x = self.__expcube.copy()
        
        if avg_replicates:
            x = self.__avg_replicates(x)
            logger.debug("The output shape is reduced to gene × condition by averaging over replicates.")
        
        if x.ndim == 3 and flatten:
            idx = np.array(self.__sample_map)
            x = x[:, idx[:,0], idx[:,1]]

        if zscore:
            x = self.__zscore(x)

        return x

    
    def __avg_replicates(self, x) -> np.ndarray:
        logger.debug("Expression data was averaged over replicates.")
        return np.nanmean(x, axis=1)
    
    
    def __zscore(self, x) -> np.ndarray:
        m = np.mean(x, axis=-1, keepdims=True)
        s = np.std(x, axis=-1, keepdims=True, ddof=0)
        s[s == 0] = 1.0
        logger.debug("Expression data was z-scored.")
        return (x - m) / s
    


    def plot(
        self,
        gene: int | str,
        file_name: str | None = None,
        figsize: tuple[float, float] = (6.0, 4.0),
        dpi: int = 300,
    ) -> None:
        """
        Plot the expression profile of one gene.

        Parameters
        ----------
        gene : int or str
            Gene row index or gene identifier.

        file_name : str or None, optional
            Output file path. If provided, the figure is saved to this path.
            If ``None``, the figure is displayed.

        figsize : tuple of float, default=(6.0, 4.0)
            Figure size in inches.

        dpi : int, default=300
            Figure resolution in dots per inch.
        """
        gene_idx = self.__genes.index(gene) if isinstance(gene, str) else gene

        # figure setup
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        x_axis = self.design("condition").unique().tolist()
        colors = [utils._COLORS[i % len(utils._COLORS)] for i in range(len(x_axis))]

        n_reps = self.__expcube.shape[1]

        for i in range(len(x_axis)):
            for r in range(n_reps):
                ax.scatter(x_axis[i], self.__expcube[gene_idx, r, i], color=colors[i], marker="o", alpha=1 / n_reps)
        ax.set_xticks(x_axis)
        ax.grid(True)

        utils._close_plt(file_name)
    


class SimExpressionData(ExpressionData):
    """
    Simulated gene expression data.

    This class extends ``ExpressionData`` by storing ground truth information
    generated during simulation, such as the cluster assignment of each gene.

    The object can be used in the same way as ``ExpressionData`` for testing,
    examples, and method evaluation.
    """
    def __init__(
        self,
        x: np.ndarray | pd.DataFrame,
        design: dict | pd.DataFrame,
        genes: Sequence[str] | None = None,
        gt: pd.DataFrame | None = None
    ) -> None:
        
        super().__init__(x, design, genes)
        self.__gt = gt


    @property
    def gt(self) -> pd.DataFrame | None:
        """
        Ground truth information for the simulated dataset.

        Returns
        -------
        pandas.DataFrame or None
            Ground truth table associated with the simulated genes. For data
            generated by ``simulate_exp_data``, this table contains gene
            identifiers and their simulated cluster labels.
        """
        return copy.deepcopy(self.__gt)


def simulate_exp_data(
    n_replicates: list[int] = [3, 3, 3, 3, 3],
    cluster_sizes: list[int] = [600, 300, 100],
    cluster_std: float = 0.35,
    replicate_noise: float = 0.15,
    seed: int | None = None,
) -> SimExpressionData:
    """
    Generate a simulated gene expression dataset.

    Parameters
    ----------
    n_replicates : list of int, default=[3, 3, 3, 3, 3]
        Number of biological replicates for each condition.

    cluster_sizes : list of int, default=[600, 300, 100]
        Number of genes assigned to each simulated expression cluster.

    cluster_std : float, default=0.35
        Standard deviation used to generate variation among genes within the
        same cluster.

    replicate_noise : float, default=0.15
        Standard deviation of random noise added to individual replicate
        measurements.

    seed : int or None, optional
        Random seed used to make the simulation reproducible.

    Returns
    -------
    SimExpressionData
        Simulated expression dataset with sample annotation and ground truth
        cluster labels.
    """
    rng = np.random.default_rng(seed)

    n_conditions = len(n_replicates)
    n_clusters = len(cluster_sizes)
    n_genes = sum(cluster_sizes)
    n_seed_genes = max(cluster_sizes) * len(cluster_sizes)
    n_total_samples = int(np.sum(n_replicates))

    condition_profile, cluster_labels = sklearn.datasets.make_blobs(
        n_samples=n_seed_genes,
        n_features=n_conditions,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=seed,
        shuffle=False,
    )
    cluster_labels = cluster_labels.astype(int)
    
    selected_indices = []
    for cl, size in enumerate(cluster_sizes):
        idx = np.where(cluster_labels == cl)[0]
        selected_indices.append(rng.choice(idx, size=size, replace=False))
    
    selected_indices = np.concatenate(selected_indices)
    condition_profile = condition_profile[selected_indices]
    cluster_labels = cluster_labels[selected_indices]
        
    x = np.zeros((n_genes, n_total_samples), dtype=float)
    design = {'condition': [], 'replicate': []}
    sample_names = []
    col = 0
    for c_idx, n_rep in enumerate(n_replicates):
        condition_name = f"C{c_idx + 1}"

        for r_idx in range(n_rep):
            replicate_name = f"R{r_idx + 1}"

            noise = rng.normal(loc=0.0, scale=replicate_noise, size=n_genes)
            x[:, col] = condition_profile[:, c_idx] + noise

            design['condition'].append(condition_name)
            design['replicate'].append(replicate_name)

            sample_names.append(f"{condition_name}_{replicate_name}")
            col += 1

    genes = [f"Gene_{i + 1:05d}" for i in range(sum(cluster_sizes))]
    design = pd.DataFrame(design, index=sample_names)
    
    gt = pd.DataFrame({
        'gene': genes,
        'cluster': cluster_labels,
    })

    return SimExpressionData(x, design, genes, gt)


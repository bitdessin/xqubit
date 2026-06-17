import logging
import numpy as np
from scipy.stats import ttest_ind, f_oneway
from . import data, qstate
logging.basicConfig()
logger = logging.getLogger(__name__)


def calc_stats(
    x: np.ndarray,
    y: np.ndarray | None = None,
    method: str = "variance",
    normalize: bool = False,
) -> np.ndarray:
    """
    Calculate gene-wise summary or group-difference scores.

    This function computes one score for each gene from a gene × sample
    expression matrix. The score can be based on overall variation, mean
    expression, or differential expression across sample groups.

    Parameters
    ----------
    x : numpy.ndarray, shape (n_genes, n_samples)
        Gene expression matrix. Rows correspond to genes and columns correspond
        to samples.

    y : numpy.ndarray or None, optional
        Group labels for samples. The length of ``y`` must match the number of
        columns in ``x``.

        This argument is required when ``method`` is ``"ttest"`` or
        ``"anova"``.

    method : {"variance", "mean", "ttest", "anova"}, default="variance"
        Scoring method.

        - ``"variance"``: variance of each gene across samples
        - ``"mean"``: mean expression level of each gene
        - ``"ttest"``: absolute Welch's t-statistic between two groups
        - ``"anova"``: one-way ANOVA F-statistic across two or more groups

    normalize : bool, default=False
        If ``True``, rescale scores to the range [0, 1].

    Returns
    -------
    numpy.ndarray, shape (n_genes,)
        Gene-wise scores.
    """
    if x.ndim != 2:
        raise ValueError("x must be a 2D array of shape gene x sample.")
    if y is not None:
        y = np.asarray(y)
        if len(y) != x.shape[1]:
            raise ValueError("Length of y must match number of samples.")

    __check_y(method, y)

    if method == "variance":
        s = np.var(x, axis=1)

    elif method == "mean":
        s = np.mean(x, axis=1)

    elif method == "ttest":
        groups = np.unique(y)
        idx1 = y == groups[0]
        idx2 = y == groups[1]

        s = np.array([
            abs(ttest_ind(x[i, idx1], x[i, idx2], equal_var=False, nan_policy="omit").statistic)
            for i in range(x.shape[0])
        ])

    elif method == "anova":
        groups = np.unique(y)

        s = np.array([
            f_oneway(*[x[i, y == g] for g in groups]).statistic
            for i in range(x.shape[0])
        ])

    else:
        raise ValueError(f"Unknown score method: {method}")

    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

    if normalize:
        den = s.max() - s.min()
        if den < 1e-12:
            return np.zeros_like(s)
        s = (s - s.min()) / den

    return s


def __check_y(method, y):
    if method in ['ttest', 'anova'] and y is None:
        raise ValueError(f"y must be provided for {method} method.")
    if method == 'ttest' and len(np.unique(y)) != 2:
        raise ValueError("ttest method requires exactly two groups in the design.")
    if method == 'anova' and len(np.unique(y)) < 2:
        raise ValueError("anova method requires at least two groups in the design.")


def calc_corrcoef(
    x: np.ndarray,
    method='pearson',
    diff: bool = False
) -> np.ndarray:
    """
    Calculate pairwise gene-gene correlation.

    Parameters
    ----------
    x : numpy.ndarray, shape (n_genes, n_conditions)
        Gene expression matrix, typically after averaging biological replicates
        for each condition or time point.

    method : {"pearson"}, default="pearson"
        Correlation method. Currently, only Pearson correlation is supported.

    diff : bool, default=False
        If ``True``, calculate correlations using changes between adjacent
        conditions or time points, defined as ``x[:, 1:] - x[:, :-1]``.

    Returns
    -------
    numpy.ndarray, shape (n_genes, n_genes)
        Symmetric matrix of pairwise gene-gene correlation coefficients.
    """
    if diff:
        x = x[:, 1:] - x[:, :-1]
    
    if method == 'pearson':
        scores = np.corrcoef(x)
    else:
        raise ValueError(f"Unknown correlation method: {method}")  
        
    return scores



def calc_cos2_similarity(
    x: np.ndarray,
    normalize: bool = False
) -> np.ndarray:
    
    """
    Calculate pairwise cosine-squared similarity between genes.

    This function compares gene expression profiles by the squared cosine of
    the angle between two expression vectors. The resulting score is high when
    two genes have similar expression profile directions, regardless of the
    sign of the cosine value.

    Parameters
    ----------
    x : numpy.ndarray, shape (n_genes, n_conditions)
        Gene expression matrix. Rows correspond to genes and columns correspond
        to conditions, time points, or other ordered measurements.

    normalize : bool, default=False
        If ``True``, rescale cosine values to the range [0, 1] before squaring.

    Returns
    -------
    numpy.ndarray, shape (n_genes, n_genes)
        Symmetric matrix of pairwise cosine-squared similarity scores.
    """
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x_norm = x / norms
    k = x_norm @ x_norm.T
    
    if normalize:
        k = (k - k.min()) / (k.max() - k.min() + 1e-8)
    
    return k**2




def calc_fidelity(
    x: np.ndarray | data.ExpressionData,
    n: int = 100,
    seed: int | None = None,
    **kwargs
) -> np.ndarray:
    """
    Calculate pairwise fidelity between gene level state vectors.

    For two normalized state vectors, fidelity is the squared absolute inner
    product between them. In this package, it is used as a gene-gene similarity
    measure after expression profiles have been encoded as normalized state
    vectors.

    Parameters
    ----------
        x : numpy.ndarray | data.ExpressionData
                Input data for fidelity computation.

                - If 2D ndarray (n_genes, n_components), interpreted as normalized
                    state vectors and fidelity is computed directly.
                - If ExpressionData, one replicate is randomly sampled per
                    condition from the 3D expression cube,
                    :func:`xqubit.qstate.build` is applied, and fidelity is
                    computed. This is repeated ``n`` times.

                3D ndarray input is not supported.

        n : int, default=100
            Number of random sampling runs for ExpressionData input.

        seed : int or None, default=None
            Random seed for reproducibility when sampling from ExpressionData.

        **kwargs
                Additional options.

                - ``seed``: int or None, random seed.
                - ``timepoints``: sequence for sampled temporary design.
                - ``conditions``: sequence for sampled temporary design.
                - ``genes``: sequence of gene names for sampled temporary data.
                - Other keyword arguments are passed to
                    :func:`xqubit.qstate.build`.

    Returns
    -------
    numpy.ndarray, shape (n_genes, n_genes)
        Symmetric matrix of pairwise fidelity scores.

        For ExpressionData input, mean and variance are computed across ``n``
        runs, and the mean matrix is returned.
    """
    def _calc_fidelity(psi):
        norm = np.sum(np.abs(psi) ** 2, axis=1)
        if not np.allclose(norm, 1.0, atol=1e-6):
            raise ValueError("Quantum states are not normalized.")
        k = psi @ psi.conj().T
        return np.abs(k) ** 2


    def _calc_fidelity_cube(x, n, seed, build_params):
        rng = np.random.default_rng(seed)
        
        exp_cube = x.exp(avg_replicates=False, zscore=False)
        condition = x.design("condition")
        timepoint = x.design("timepoint")
        n_genes, _, n_conditions = exp_cube.shape

        valid_replicates = []
        for c in range(n_conditions):
            valid = np.where(~np.isnan(exp_cube[:, :, c]).any(axis=0))[0]
            if valid.size == 0:
                raise ValueError(f"No valid replicate without NaN is available for condition index {c}.")
            valid_replicates.append(valid)

        f_sum = np.zeros((n_genes, n_genes), dtype=float)
        for i in range(n):
            rep_indices = np.array([rng.choice(valid_replicates[c]) for c in range(n_conditions)])
            exp_mat = exp_cube[:, rep_indices, np.arange(n_conditions)]

            sampled_data = data.ExpressionData(
                exp_mat,
                {
                    "condition": condition,
                    "replicate": np.ones(n_conditions, dtype=int),
                    "timepoint": timepoint,
                }
            )
            built = qstate.build(sampled_data, **build_params)
            psi = built["statevector"] if isinstance(built, dict) else built
            f_sum += _calc_fidelity(psi)

        f_mat = f_sum / n
        return f_mat


    if isinstance(x, np.ndarray):
        if x.ndim == 2:
            return _calc_fidelity(x)
        else:
            raise ValueError("x ndarray must be 2D.")
    elif isinstance(x, data.ExpressionData):
        return _calc_fidelity_cube(x, n, seed, kwargs)
    else:
        raise TypeError("x must be a numpy.ndarray or ExpressionData.")




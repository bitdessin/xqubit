import logging
import math
import random
from typing import Literal
import joblib

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
import tqdm
import itertools

logging.basicConfig()
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from . import utils


def build_network(
    x: np.ndarray,
    s: float = 0.5,
    k: int = 10,
    mutual_knn: bool = True,
    seed=None,
) -> ig.Graph:
    """
    Build a sparse weighted gene network from a similarity matrix.


    This function converts a gene-gene similarity matrix into an undirected
    weighted graph. Edges are selected by applying a similarity threshold,
    retaining up to ``k`` nearest neighbors for each gene, and optionally
    keeping only mutual nearest-neighbor relationships.

    Parameters
    ----------
    x : numpy.ndarray, shape (n_genes, n_genes)
        Symmetric gene-gene similarity matrix. Diagonal values are ignored.

    s : float, default=0.5
        Similarity cutoff for edge selection.

        - If ``0 <= s <= 1``, ``s`` is used as an absolute similarity cutoff.
        - If ``s > 1``, ``s`` is interpreted as a percentile of the
          upper-triangular similarity values.

    k : int, default=10
        Maximum number of neighbors retained for each gene after thresholding.

    mutual_knn : bool, default=True
        If ``True``, keep an edge only when both genes select each other as
        neighbors. If ``False``, keep an edge when either gene selects the
        other.

    seed : int or None, optional
        Random seed.

    Returns
    -------
    igraph.Graph
        Weighted undirected graph. Edge weights are stored in
        ``g.es["weight"]``.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    n_nodes = x.shape[0]
    k = int(k)

    # score cutoff
    if s > 1.0:
        s = np.percentile(x[np.triu_indices(n_nodes, k=1)], s)

    adj = np.zeros_like(x, dtype=np.float64)
    for i in range(n_nodes):
        candidates = np.where((x[i] >= s) & (np.arange(n_nodes) != i))[0]
        if candidates.size == 0:
            continue

        order = np.argsort(x[i, candidates])
        topk = candidates[order[-k:]] if candidates.size > k else candidates

        adj[i, topk] = x[i, topk]

    # mutual or non-mutual kNN
    adj = np.minimum(adj, adj.T) if mutual_knn else np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0.0)

    i_idx, j_idx = np.where(np.triu(adj, 1) > 0)

    g = ig.Graph()
    g.add_vertices(n_nodes)
    g.add_edges(list(zip(i_idx, j_idx)))
    g.es["weight"] = adj[i_idx, j_idx]

    logger.info(f"Network built with {s=}, {k=}, {mutual_knn=}.")

    return g



def detect_communities(
    g: ig.Graph,
    min_size: int = 20,
    resolution: float = 1.0,
    n_iterations: int = 100,
    consensus_threshold: float = 0.8,
    seed=None,
    format: Literal["list", "partition"] = "list",
) -> list[int] | leidenalg.RBConfigurationVertexPartition:
    """
    Detect gene communities using Leiden clustering.


    This function detects communities in a weighted gene network. When
    ``n_iterations`` is greater than 1, Leiden clustering is repeated multiple
    times, a consensus network is built from co-clustering frequencies, and a
    final Leiden clustering is performed on the consensus network.

    Parameters
    ----------
    g : igraph.Graph
        Weighted undirected graph. Edge weights must be stored in
        ``g.es["weight"]``.

    min_size : int, default=100
        Minimum size of reported communities. Communities smaller than this
        value are assigned to community 0.

    resolution : float, default=1.0
        Resolution parameter for Leiden clustering. Larger values usually
        produce more and smaller communities.

    n_iterations : int, default=100
        Number of Leiden runs used to build the consensus network. If set to
        1, consensus clustering is skipped.

    consensus_threshold : float, default=0.8
        Threshold for retaining edges in the consensus network.

        - If ``0 <= consensus_threshold <= 1``, it is used as an absolute
          co-clustering frequency cutoff.
        - If ``consensus_threshold > 1``, it is interpreted as a percentile of
          upper-triangular consensus values.

    seed : int or None, optional
        Random seed. When provided, each repeated run uses a different
        deterministic seed.

    format : {"list", "partition"}, default="list"
        Output format.

        - ``"list"``: return one community label per gene
        - ``"partition"``: return the Leiden partition object

    Returns
    -------
    list of int or leidenalg.RBConfigurationVertexPartition
        Community labels or a Leiden partition object, depending on
        ``format``.
    """
    if format not in ("list", "partition"):
        raise ValueError(f"Invalid format: {format}, must be 'list' or 'partition'")

    n_nodes = g.vcount()
    partition = None
    memberships = []

    # bootstrap Leiden runs
    logger.debug(f"Running Leiden community detection {n_iterations} times with resolution={resolution} ...")
    for t in tqdm.tqdm(range(n_iterations), desc="Repeating community detection", leave=False, disable=(n_iterations <= 1)):
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=None if seed is None else seed + t,
        )
        memberships.append(partition.membership)

    if n_iterations > 1:
        # build consensus matrix
        logger.debug(f"Building consensus matrix from {n_iterations} Leiden runs...")
        consensus_mat = np.zeros((n_nodes, n_nodes), dtype=float)
        for membership in memberships:
            m = np.array(membership)
            consensus_mat += m[:, None] == m[None, :]
        consensus_mat /= n_iterations

        if consensus_threshold > 1.0:
            consensus_threshold = np.percentile(consensus_mat[np.triu_indices(n_nodes, k=1)], consensus_threshold)
        consensus_mat[consensus_mat < consensus_threshold] = 0.0

        # build consensus graph
        logger.debug(f"Building consensus graph ...")
        i_idx, j_idx = np.where(np.triu(consensus_mat, 1) > 0)

        g_consensus = ig.Graph()
        g_consensus.add_vertices(n_nodes)
        g_consensus.add_edges(list(zip(i_idx, j_idx)))
        g_consensus.es["weight"] = consensus_mat[i_idx, j_idx]

        # final Leiden on consensus
        logger.debug(f"Running final Leiden on consensus graph ...")
        partition = leidenalg.find_partition(
            g_consensus,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=None if seed is None else seed + n_iterations + 1,
        )
    
    logger.debug(f'Communities detected with resolution={resolution}, n_iterations={n_iterations}, consensus_threshold={consensus_threshold}.')

    if format == "partition":
        return partition
    else:
        return _get_communities_label(partition, min_size, "label", seed)


def _get_communities_label(partition, min_size, format, seed):
    logger.debug(f"Summarizing communities and pooling small communities into one ...")

    comm_0 = []
    comm_l = []
    comm_id = 1
    for comm in partition:
        comm_id += 1
        if len(comm) < min_size:
            comm_0.extend(comm)
        else:        
            comm_l.append(np.array(comm))
    
    comm_l.sort(key=len, reverse=True)
    comms = [comm_0] + comm_l
    
    logger.debug(f"Detected {len(partition)} communities before post-processing.")
    logger.debug(f"Detected {len(comms) - 1} communities and 1 outlier community ({len(comm_0)} genes) after pooling small communities and subclustering.")
    logger.info(f"Detected {len(comms) - 1} communities (+ {len(comm_0)} outlier genes)")

    if format == "community_list":
        return comms

    elif format == "label":
        cl = np.zeros(len(partition.graph.vs), dtype=int)
        for cid, comm in enumerate(comms):
            cl[list(comm)] = cid
        return cl.tolist()

    else:
        raise ValueError(f"Invalid format: {format}, must be 'community_list' or 'label'")


def _subcluster_comms(x, cutoff, seed):
    x = np.asarray(x)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    kmeans = KMeans(n_clusters=2, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(x)
    score = silhouette_score(x, labels)
    
    return labels if score > cutoff else None


def save_communities(
    x: list[int],
    file_name: str,
    nodes: list[str] | None = None
) -> None:
    """
    Save gene community assignments to a tab-separated file.

    Parameters
    ----------
    x : list of int
        Community label for each gene or node. Community 0 is used for genes
        assigned to small communities that were pooled during post-processing.

    file_name : str
        Output file path.

    nodes : list of str or None, optional
        Gene or node identifiers corresponding to ``x``. If ``None``, integer
        node indices are written.
    """
    if nodes is not None and len(nodes) != len(x):
        raise ValueError("Length of nodes must match length of x")
    
    with open(file_name, "w") as f:
        f.write("gene\tcommunity\n")
        for gid, cid in enumerate(x):
            gene = nodes[gid] if nodes is not None else gid
            f.write(f"{gene}\t{cid}\n")


def plot_communities(
    x,
    exp: np.ndarray,
    x_labels: np.ndarray | None = None,
    i: int | None = None,
    file_name: str | None = None,
    figsize: float | tuple[float, float] = (4, 4),
    line_width: float = 1.0,
    alpha: float = 0.1,
    ylim: tuple[float, float] | None = None,
) -> None:    
    """
    Plot expression profiles for genes grouped by community.

    Parameters
    ----------
    x : sequence of int
        Community label for each gene. The length must match the number of
        rows in ``exp``.

    exp : numpy.ndarray, shape (n_genes, n_conditions)
        Gene expression matrix to plot.

    x_labels : numpy.ndarray or None, optional
        Labels or values shown on the x-axis. If ``None``, column indices of
        ``exp`` are used.

    i : int or None, optional
        Community to plot. If ``None``, all communities except community 0 are
        plotted.

    file_name : str or None, optional
        Output file path. If provided, the figure is saved to this path.
        If ``None``, the figure is displayed.

    figsize : float or tuple of float, default=(4, 4)
        Size of each subplot in inches. If a single number is given, the same
        value is used for width and height.

    line_width : float, default=1.0
        Line width for individual gene expression profiles.

    alpha : float, default=0.1
        Transparency of individual gene expression profiles.

    ylim : tuple of float or None, optional
        Limits of the y-axis. If ``None``, limits are determined automatically.
    """
    if exp.ndim != 2:
        raise ValueError("exp must be 2D array.")
    if len(x) != exp.shape[0]:
        raise ValueError("Length of x must match number of rows in exp.")

    n_comm = len(set(x))

    if x_labels is None:
        x_labels = np.arange(exp.shape[1])

    if i is not None:
        if i < 0 or i >= n_comm:
            raise ValueError(f"Invalid community index: {i}")
        comm_ids = [i]
    else:
        comm_ids = [cid for cid in range(n_comm) if cid != 0]
        
    if isinstance(figsize, (int, float)):
        figsize = (figsize, figsize)

    n_plots = len(comm_ids)
    if n_plots < 1:
        return None
    elif n_plots == 1:
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        axes = [axes]
    else:
        ncols = math.ceil(math.sqrt(n_plots))
        nrows = math.ceil(n_plots / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows), squeeze=False)
        axes = axes.flatten()

    for ax, cid in zip(axes, comm_ids):
        comm_genes = [gid for gid, label in enumerate(x) if label == cid]

        if len(comm_genes) == 0:
            ax.set_title(f"Community {cid} (empty)")
            ax.axis("off")
            continue

        comm_genes_exp = exp[comm_genes, :]

        for g in range(comm_genes_exp.shape[0]):
            ax.plot(x_labels, comm_genes_exp[g], linewidth=line_width, alpha=alpha, color="#333333",)
            if ylim is not None:
                ax.set_ylim(ylim)

        ax.set_title(f"Community {cid} (n={len(comm_genes)})")

        ax.grid(alpha=0.3)

    for ax in axes[n_plots:]:
        ax.axis("off")

    utils._close_plt(file_name)


def scan_network_params(
    x: np.ndarray,
    s_cutoffs: list[float] | np.ndarray = [50, 60, 70, 80, 85, 90, 95],
    k_cutoffs: list[int] | np.ndarray = np.arange(5, 31, 5),
    resolutions: float | np.ndarray = np.logspace(-0.1, 0.3, 5),
    min_size: int = 20,
    n_iterations: int = 100,
    mutual_knn: bool = True,
    n_threads: int = 1,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Scan network construction and community detection parameters.

    This function evaluates combinations of similarity cutoffs, k-nearest
    neighbor settings, and Leiden resolution parameters. For each parameter
    set, it builds a gene network, repeats community detection, and summarizes
    network modularity, clustering stability, gene coverage, and community
    size statistics.

    Parameters
    ----------
    x : numpy.ndarray, shape (n_genes, n_genes)
        Symmetric gene-gene similarity matrix.

    s_cutoffs : list of float or numpy.ndarray, default=[50, 60, 70, 80, 85, 90, 95]
        Similarity cutoffs passed to ``build_network``. Values greater than 1
        are interpreted as percentiles of the upper-triangular similarity
        values.

    k_cutoffs : list of int or numpy.ndarray, default=np.arange(5, 31, 5)
        Values of ``k`` passed to ``build_network``.

    resolutions : float, list of float, or numpy.ndarray, default=np.logspace(-0.1, 0.3, 5)
        Leiden resolution parameters to evaluate.

    min_size : int, default=100
        Minimum size of reported communities. Smaller communities are pooled
        into community 0.

    n_iterations : int, default=100
        Number of repeated Leiden runs for each parameter set.

    mutual_knn : bool, default=True
        Whether to use mutual k-nearest-neighbor filtering when building the
        network.

    n_threads : int, default=1
        Number of parallel jobs.

    seed : int or None, optional
        Random seed.

    Returns
    -------
    pandas.DataFrame
        Parameter scan results. Each row corresponds to one parameter
        combination and contains summary statistics for modularity, clustering
        stability, gene coverage, and community sizes.
    """
    param_grid = list(enumerate(itertools.product(s_cutoffs, k_cutoffs, resolutions)))
    
    scan_records = joblib.Parallel(n_jobs=n_threads)(
        joblib.delayed(_scan_network_params)(x, s, k, resolution, min_size, n_iterations, mutual_knn,
                                             seed + i if seed is not None else None)
            for i, (s, k, resolution) in tqdm.tqdm(param_grid)
    )
    return pd.DataFrame(scan_records)


def _scan_network_params(x, s, k, resolution, min_size, n_iterations, mutual_knn, seed):
    g = build_network(x, s, k, mutual_knn=mutual_knn, seed=seed)
    scan_record = {
        "modularity": [],
        "label": [],
        "ncomm": [],
        "coverage": [],
        "meancommsize": [],
        "mediancommsize": [],
        'maxcommsize': [],
        'mincommsize': [],
    }

    for r in range(n_iterations):
        seed_r = seed
        if seed is not None:
            seed_r = seed * n_iterations + r
            random.seed(seed_r)
            np.random.seed(seed_r)

        part = detect_communities(g, min_size=min_size, resolution=resolution, n_iterations=1, seed=seed_r, format="partition")

        # modularity
        scan_record["modularity"].append(part.modularity)

        # labels for stability
        labels = np.zeros(g.vcount(), dtype=int)
        for cid, comm in enumerate(part):
            labels[list(comm)] = cid
        scan_record["label"].append(labels)

        # community data
        comms = _get_communities_label(part, min_size, "community_list", seed_r)
        

        comm_sizes = [len(genes) for cid, genes in enumerate(comms) if cid != 0]
        if len(comm_sizes) > 0:
            scan_record["ncomm"].append(len(comm_sizes))
            scan_record["coverage"].append(np.sum(comm_sizes))
            scan_record["meancommsize"].append(np.mean(comm_sizes))
            scan_record["mediancommsize"].append(np.median(comm_sizes))
            scan_record['maxcommsize'].append(np.max(comm_sizes))
            scan_record['mincommsize'].append(np.min(comm_sizes))
            
    # stability (NMI, AMI)
    nmi_vals = []
    ami_vals = []
    for i in range(len(scan_record["label"])):
        for j in range(i + 1, len(scan_record["label"])):
            labels_i = scan_record["label"][i]
            labels_j = scan_record["label"][j]
            mask = (labels_i != 0) & (labels_j != 0)
            if np.sum(mask) > 1:
                nmi_vals.append(normalized_mutual_info_score(labels_i[mask], labels_j[mask]))
                ami_vals.append(adjusted_mutual_info_score(labels_i[mask], labels_j[mask]))

    s_abs = s if s <= 1.0 else np.percentile(x[np.triu_indices(x.shape[0], k=1)], s)
    gs_stats = {'s': s, 's_abs': s_abs, 'k': k, 'resolution': resolution}
    # module stats
    _summarise_stats(gs_stats, 'modularity', scan_record["modularity"])
    _summarise_stats(gs_stats, 'nmi', nmi_vals)
    _summarise_stats(gs_stats, 'ami', ami_vals)
    # community size stats
    _summarise_stats(gs_stats, 'genecoverage', scan_record["coverage"])
    _summarise_stats(gs_stats, 'ncomm', scan_record["ncomm"])
    _summarise_stats(gs_stats, 'meancommsize', scan_record["meancommsize"])
    _summarise_stats(gs_stats, 'mediancommsize', scan_record["mediancommsize"])
    _summarise_stats(gs_stats, 'maxcommsize', scan_record["maxcommsize"])
    _summarise_stats(gs_stats, 'mincommsize', scan_record["mincommsize"])
    
    return gs_stats


def _summarise_stats(d: dict, k: str, x) -> None:
    v = x if x is not None and len(x) > 0 else np.nan
    d[f'{k}_mean'] = np.mean(v)
    d[f'{k}_std'] = np.std(v)


def rank_network_params(
    df: pd.DataFrame,
    opt_vars: dict | None = None,
    balance_weights: dict | None = None,
) -> pd.DataFrame:
    """
    Rank network parameter sets from a parameter scan table.

    This function filters parameter sets by acceptable value ranges, identifies
    Pareto-optimal solutions among selected objective variables, and selects
    one balanced solution from the Pareto front.

    Parameters
    ----------
    df : pandas.DataFrame
        Output table from ``scan_network_params``.

    opt_vars : dict or None, optional
        Filtering and optimization criteria.

        Each key is a column name in ``df``. Each value is a dictionary that
        can contain:

        - ``"min"``: minimum acceptable value
        - ``"max"``: maximum acceptable value
        - ``"direction"``: optimization direction, either ``"max"`` or
          ``"min"``

        Variables with ``"min"`` or ``"max"`` are used for filtering.
        Variables with ``"direction"`` are also used for Pareto-front
        detection and balanced-solution selection.

    balance_weights : dict or None, optional
        Weights used to select the balanced solution from the Pareto front.
        Keys must match variables in ``opt_vars`` that have ``"direction"``.
        If ``None``, all objective variables are weighted equally.

    Returns
    -------
    pandas.DataFrame
        Ranked parameter table with additional columns:

        - ``score``: ranking score
        - ``within_opt_range``: whether the parameter set passed range filters
        - ``is_pareto``: whether the parameter set is Pareto-optimal
        - ``is_balanced``: whether the parameter set is the selected balanced
          solution
        - ``balance_distance``: distance from the normalized ideal point
        - ``balance_rank``: rank by ``balance_distance`` among Pareto-optimal
          solutions

    Notes
    -----
    The ``score`` column is coded as follows:

    - ``0``: outside the acceptable range
    - ``1``: within the acceptable range
    - ``2``: Pareto-optimal
    - ``3``: selected balanced solution
    """
    if opt_vars is None:
        opt_vars = {
            "modularity_mean": {"min": 0.4},
            "ami_mean": {"min": 0.7},
            "genecoverage_mean": {"min": 1000},
            "ncomm_mean": {"min": 0, "direction": "min"},
            "mediancommsize_mean": {"min": 20, "direction": "max"},
        }

    df = df.copy()

    df["score"] = 0
    df["within_opt_range"] = False
    df["is_pareto"] = False
    df["is_balanced"] = False
    df["balance_distance"] = np.nan
    df["balance_rank"] = np.nan

    # validate opt_vars
    for var, cond in opt_vars.items():
        if var not in df.columns:
            raise ValueError(f"{var} not found in DataFrame columns")

        vmin = cond.get("min", -np.inf)
        vmax = cond.get("max", np.inf)

        if vmin > vmax:
            raise ValueError(
                f"Invalid range for {var}: min={vmin} is greater than max={vmax}"
            )

        direction = cond.get("direction", None)

        if direction is not None and direction not in {"max", "min"}:
            raise ValueError(
                f"Invalid direction for {var}: {direction}. "
                "Expected 'max', 'min', or no direction."
            )

    # variables used only for filtering
    filter_vars = list(opt_vars.keys())

    # variables used for Pareto front and balanced solution
    objective_vars = [
        var for var, cond in opt_vars.items()
        if cond.get("direction", None) in {"max", "min"}
    ]

    # filtering based on all opt_vars ranges
    valid_mask = np.ones(len(df), dtype=bool)

    for var in filter_vars:
        cond = opt_vars[var]

        vmin = cond.get("min", -np.inf)
        vmax = cond.get("max", np.inf)

        valid_mask &= df[var].notna().to_numpy()
        valid_mask &= (df[var] >= vmin).to_numpy()
        valid_mask &= (df[var] <= vmax).to_numpy()

    df.loc[valid_mask, "within_opt_range"] = True
    df.loc[valid_mask, "score"] = 1

    df_candidates = df.loc[valid_mask].copy()

    if df_candidates.empty:
        return df.sort_values(
            by=["score"],
            ascending=[False],
            na_position="last",
        )

    # If no optimization variables are specified, stop after filtering.
    if len(objective_vars) == 0:
        return df.sort_values(
            by=["score"],
            ascending=[False],
            na_position="last",
        )

    # Pareto front detection among filtered candidates
    values = df_candidates[objective_vars].to_numpy(dtype=float)

    # Convert all objectives to maximization.
    # For direction == "min", multiply by -1.
    objective_values = values.copy()

    for j, var in enumerate(objective_vars):
        direction = opt_vars[var]["direction"]

        if direction == "min":
            objective_values[:, j] *= -1.0

    n = len(objective_values)
    pareto_mask = np.ones(n, dtype=bool)

    for i in range(n):
        if not pareto_mask[i]:
            continue

        for j in range(n):
            if i == j:
                continue

            j_better_or_equal = np.all(objective_values[j] >= objective_values[i])
            j_strictly_better = np.any(objective_values[j] > objective_values[i])

            if j_better_or_equal and j_strictly_better:
                pareto_mask[i] = False
                break

    pareto_indices = df_candidates.index[pareto_mask]

    df.loc[pareto_indices, "is_pareto"] = True
    df.loc[pareto_indices, "score"] = 2

    # balanced solution selection from Pareto front
    df_pareto = df.loc[pareto_indices].copy()

    if not df_pareto.empty:
        if balance_weights is None:
            balance_weights = {var: 1.0 for var in objective_vars}
        else:
            missing_weights = set(objective_vars) - set(balance_weights.keys())
            if missing_weights:
                raise ValueError(
                    f"balance_weights is missing weights for: {missing_weights}"
                )

        weights = np.array([float(balance_weights[var]) for var in objective_vars])

        if np.any(weights < 0):
            raise ValueError("All balance_weights must be non-negative")

        if np.all(weights == 0):
            raise ValueError("At least one balance weight must be positive")

        norm_values = np.zeros((len(df_pareto), len(objective_vars)), dtype=float)

        for j, var in enumerate(objective_vars):
            x = df_pareto[var].to_numpy(dtype=float)

            cond = opt_vars[var]
            direction = cond["direction"]

            vmin = cond.get("min", -np.inf)
            vmax = cond.get("max", np.inf)

            # Prefer opt_vars range for normalization.
            # If bounds are not finite or degenerate, fall back to observed Pareto range.
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin = np.nanmin(x)
                vmax = np.nanmax(x)

            if vmax == vmin:
                # If there is no variation, this variable does not distinguish
                # Pareto solutions. Set it to ideal for all rows.
                norm = np.ones_like(x, dtype=float)
            else:
                if direction == "max":
                    norm = (x - vmin) / (vmax - vmin)
                elif direction == "min":
                    norm = (vmax - x) / (vmax - vmin)
                else:
                    raise ValueError(f"Invalid direction for {var}: {direction}")

                norm = np.clip(norm, 0.0, 1.0)

            norm_values[:, j] = norm

        # Distance to ideal point: ideal = 1 for all normalized objectives.
        balance_distance = np.sqrt(
            np.average((1.0 - norm_values) ** 2, axis=1, weights=weights)
        )

        df.loc[df_pareto.index, "balance_distance"] = balance_distance

        balance_rank = (
            pd.Series(balance_distance, index=df_pareto.index)
            .rank(method="first", ascending=True)
        )

        df.loc[df_pareto.index, "balance_rank"] = balance_rank

        balanced_index = balance_rank.idxmin()

        df.loc[balanced_index, "is_balanced"] = True
        df.loc[balanced_index, "score"] = 3

    # sorting
    sort_cols = ["balance_rank", "score"]
    ascending = [True, False]

    for var in objective_vars:
        cond = opt_vars[var]
        sort_cols.append(var)
        ascending.append(cond["direction"] == "min")

    filter_only_vars = [
        var for var in filter_vars
        if var not in objective_vars
    ]

    for var in filter_only_vars:
        sort_cols.append(var)
        ascending.append(True)

    return df.sort_values(
        by=sort_cols,
        ascending=ascending,
        na_position="last",
    )


def plot_network_params(
    df: pd.DataFrame,
    xlabel: str = "ami_mean",
    ylabel: str = "modularity_mean",
    file_name: str | None = None,
) -> None:
    """
    Plot network parameter scan results.

    ```
    This function creates an interactive scatter plot from a ranked parameter
    scan table. Points are grouped by ``score`` so that acceptable,
    Pareto-optimal, and selected balanced parameter sets can be inspected.

    Parameters
    ----------
    df : pandas.DataFrame
        Output table from ``rank_network_params``.

    xlabel : str, default="ami_mean"
        Column name used for the x-axis.

    ylabel : str, default="modularity_mean"
        Column name used for the y-axis.

    file_name : str or None, optional
        Output HTML file path. If provided, the plot is saved to this path.
        If ``None``, the plot is displayed.

    Returns
    -------
    None
        This function is used for visualization and does not return a value.
    """
    def make_hover_text(df_):
        texts = []
        for _, r in df_.iterrows():
            lines = [f"{c}: {r[c]}" for c in hover_cols]
            texts.append("<br>".join(lines))
        return texts

    hover_cols = df.columns.tolist()

    fig = go.Figure()
    for score in sorted(df["score"].unique()):
        df_s = df[df["score"] == score]
        if df_s.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=df_s[xlabel],
                y=df_s[ylabel],
                mode="markers",
                name=f"score={score}",
                marker=dict(
                    size=10 + 5 * score,
                    color=utils._COLORS[score],
                    line=dict(width=1 if score >= 2 else 0, color="black"),
                    opacity=0.85 if score > 0 else 0.4,
                ),
                hoverinfo="text",
                text=make_hover_text(df_s),
            )
        )

    fig.update_layout(
        title=f"Network Parameter Scan Results",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template="simple_white",
        width=1000,
        height=800,
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
        ),
    )

    fig.show() if file_name is None else fig.write_html(file_name)

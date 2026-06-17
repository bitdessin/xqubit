import logging
import copy
import numpy as np
import matplotlib.pyplot as plt

from . import utils
from .data import ExpressionData

logging.basicConfig()
logger = logging.getLogger(__name__)

ALLOWED_ENCODINGS = ["EA",
                     "TDP", "UATDP", "ODTDP", "UAODTDP",
                     "IDP", "UAIDP", "ODIDP", "UAODIDP"]


def build(
    x: ExpressionData,
    encoding: str = 'TDP',
    alpha: float | None = None,
    alpha_scale: float = 1.0,
    weights: str | list[str] | None = 'amp',
    output: 'str' = 'statevector',
) -> np.ndarray | dict: 
    """    
    Build state-vector representations from expression profiles.

    This function converts each gene expression profile into a normalized
    complex-valued state vector. These state vectors can then be used to
    calculate gene-gene similarity by fidelity or by SWAP test circuits.

    Parameters
    ----------
    x : ExpressionData
        Gene expression data. The design table must contain a ``timepoint``
        column when temporal encodings are used.

    encoding : str, default="TDP"
        Encoding method.

        Supported values are:

        - ``"EA"``: expression amplitude encoding
        - ``"TDP"``: temporal-difference phase encoding
        - ``"IDP"``: integrated-difference phase encoding
        - ``"ODTDP"``: TDP with orthogonalized direction branches
        - ``"ODIDP"``: IDP with orthogonalized direction branches

    alpha : float or None, optional
        Phase scaling parameter. If ``None``, ``alpha`` is determined
        automatically from the distribution of temporal features.

    alpha_scale : float, default=1.0
        Multiplicative factor applied to ``alpha``. Larger values increase
        phase separation between genes.

    weights : {"amp", "phase"}, list of {"amp", "phase"}, or None, optional
        Optional weighting for unequally spaced time points.

        - ``"amp"``: weight amplitudes by interval length
        - ``"phase"``: weight phase differences by inverse interval length

        Weighting is ignored for ``"EA"``.

    output : {"statevector", "full"}, default="statevector"
        Output format.

        - ``"statevector"``: return only the normalized state vectors
        - ``"full"``: return state vectors and intermediate components

    Returns
    -------
    numpy.ndarray or dict
        If ``output="statevector"``, returns a complex-valued array with shape
        (n_genes, n_components).

        If ``output="full"``, returns a dictionary containing:

        - ``"amplitude"``: amplitude component
        - ``"phase"``: phase component
        - ``"statevector"``: normalized state vectors
        - ``"x"``: expression array used for encoding
        - ``"z"``: z-scored expression array used for encoding
        - ``"weights"``: amplitude and phase weights used for construction
    """
    if encoding not in ALLOWED_ENCODINGS:
        raise ValueError(f'Unknown encoding: {encoding}')
    if output not in ['statevector', 'full']:
        raise ValueError(f'Unknown output type: {output}')
    
    weights = __validate_weights(weights)
    
    if encoding == 'EA' and len(weights) > 0:
        logger.warning(f"Weighting is not applicable for EA encoding. Ignoring weights: {weights}.")
        weights = set()
    
    logger.debug(f'Start to build quantum state with {encoding=} ({weights=}) and {alpha=} and {alpha_scale=}.')
    
    # gene expression data for quantum state construction ()
    replicate_mode = 'mean' # if not mean, build for each replicate.
    x_mat = x.exp(avg_replicates=(replicate_mode == 'mean'), zscore=False)
    z_mat = x.exp(avg_replicates=(replicate_mode == 'mean'), zscore=True)
    
    # weights for amplitude and phase components
    dt = np.diff(list(x.design('timepoint')))
    if np.any(dt <= 0):
        raise ValueError("Ensure that time points are strictly increasing for temporal encoding.")
    amp_weight = np.sqrt(dt / np.sum(dt))[None, :] if 'amp' in weights else 1.0
    phase_weight = (1.0 / dt)[None, :] if 'phase' in weights else 1.0

    # build quantum state components
    amp = __build_qstate_amp(x_mat, z_mat, encoding, amp_weight)
    phase, alpha = __build_qstate_phase(x_mat, z_mat, encoding, alpha, alpha_scale, phase_weight)
    
    if 'OD' in encoding:
        amp, phase = __bisect_direction(x_mat, z_mat, encoding, amp, phase, phase_weight)
        logger.debug(f"Applied orthogonalization by bisecting the state space based on expression change directions.")
    
    psi = amp * phase
    __validate_qstate_norm(amp, phase, psi)
    
    logger.debug(f"Quantum state building was completed with the shape of ({psi.shape}) and scale ({alpha_scale} * {alpha}).")

    if output == "statevector":
        return psi
    elif output == "full":
        return {
            "amplitude": amp,
            "phase": phase,
            "statevector": psi,
            "x": x_mat,
            "z": z_mat,
            'weights': {'amp': amp_weight, 'phase': phase_weight},
        }



def __build_qstate_amp(x, z, encoding, weights):
    n_conditions = x.shape[-1]

    if encoding[:2] == 'UA':
        # uniform amplitude
        n_components = n_conditions if encoding == 'EA' else n_conditions - 1
        amp = np.full(x.shape[:-1] + (n_components,), 1 / np.sqrt(n_components), dtype=np.float64)
    
    else:
        # expression amplitude
        if encoding == "EA":
            amp = x
        else:
            # interval average for TPD/IDP/ODTDP/ODIDP
            amp = 0.5 * (x[..., :-1] + x[..., 1:])
    
    if np.any(amp < 0):
        raise ValueError("Negative expression values detected. Amplitude encoding requires non-negative values.")
    
    amp = amp * weights
       
    amp_norm = np.linalg.norm(amp, axis=-1, keepdims=True)
    amp_norm = np.where(amp_norm == 0, 1.0, amp_norm)
    amp = amp / amp_norm
        
    return amp


def __build_qstate_phase(x, z, encoding, alpha, alpha_scale, weights, pcut_base = 95.0):
    phase = None
    dz = __diff(z) * weights
        
    if encoding == "EA":
        if alpha is None:
            alpha = (np.pi / 2) / (x.shape[-1] - 1)
        theta = np.arange(x.shape[-1]) * alpha * alpha_scale
        __valid_phase_range(theta, np.pi * 2, pcut_base)
        phase = np.exp(1j * theta).astype(np.complex128)
        phase = np.broadcast_to(phase.reshape((1,) * (x.ndim - 1) + (x.shape[-1],)), x.shape).copy()

    elif 'TDP' in encoding:
        if alpha is None:
            dz_clipped = __clip(dz, pcut_base, False)
            alpha = (np.pi / 4) / np.max(np.abs(dz_clipped))
        theta = dz * alpha * alpha_scale
        __valid_phase_range(theta, np.pi * 2, pcut_base)
        phase = np.exp(1j * theta).astype(np.complex128)

    elif 'IDP' in encoding:
        sigma = np.std(dz, ddof=1)
        if (not np.isfinite(sigma)) or (sigma == 0):
            sigma = 1.0
        omega = dz / sigma
        phi = np.cumsum(omega, axis=-1)
        
        if alpha is None:
            phi_clipped = __clip(phi, pcut_base, True)
            alpha = (np.pi / 4) / np.max(np.abs(phi_clipped))

        theta = phi * alpha * alpha_scale
        __valid_phase_range(theta, np.pi * 2, pcut_base)
        
        phase = np.exp(1j * theta).astype(np.complex128)

    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    return phase, alpha



def __bisect_direction(x, z, encoding, amp, phase, weights, pcut_base = 95.0, assign_mode='tanh', kappa = 1.0):   
    n_intervals = amp.shape[-1]
    dz = __diff(z) * weights

    assert amp.shape == phase.shape, f"Amplitude and phase must have the same shape, but got amp={amp.shape} and phase={phase.shape}."
    assert amp.shape == dz.shape, f"Amplitude and dz must have the same shape, but got amp={amp.shape} and dz={dz.shape}."

    # bisect the state into two branches based on the sign of expression changes
    if assign_mode == 'hard':
        q_plus = np.zeros_like(dz, dtype=np.float64)
        q_minus = np.zeros_like(dz, dtype=np.float64)
        q_plus[dz > 0] = 1.0
        q_minus[dz < 0] = 1.0
        q_plus[dz == 0] = 0.5
        q_minus[dz == 0] = 0.5
    elif assign_mode == 'tanh':
        q_plus = 0.5 * (1.0 + np.tanh(kappa * dz))
        q_minus = 1.0 - q_plus
    
    # update amp/phase
    amp_od = np.zeros(amp.shape[:-1] + (2 * n_intervals,), dtype=np.float64)
    phase_od = np.zeros(phase.shape[:-1] + (2 * n_intervals,), dtype=np.complex128)
    amp_od[..., 0::2] = amp * np.sqrt(q_plus)
    amp_od[..., 1::2] = amp * np.sqrt(q_minus)
    phase_od[..., 0::2] = phase
    phase_od[..., 1::2] = phase

    return amp_od, phase_od


def __diff(x):
    return np.diff(x, axis=-1)

    
def __clip(x, pcut_base=95.0, use_row_max=False):    
    if use_row_max:
        v = np.max(np.abs(x), axis=-1)
    else:
        v = np.abs(x).ravel()
    
    clip_cutoff = np.percentile(v, pcut_base)
    if clip_cutoff == 0 or not np.isfinite(clip_cutoff):
        logger.debug(f"Non-finite or zero cutoff detected. Setting cutoff to 1.0.")
        clip_cutoff = 1.0
    
    x = np.clip(x, -clip_cutoff, clip_cutoff)
    return x
    


def __valid_phase_range(x, max_threshold, pcut_base):
    up_val = np.percentile(x, pcut_base)
    lw_val = np.percentile(x, 100 - pcut_base)
    logger.debug(f"Phase range: min={lw_val:.3f}, max={up_val:.3f}")
    if max(abs(up_val), abs(lw_val)) > max_threshold:
        logger.debug("Phase range exceeds threshold.")
        logger.warning(f"Phases are ranged between min={lw_val:.3f}, max={up_val:.3f}, resulting phase overlaps. Consider reducing alpha or alpha_scale.")


def __validate_qstate_norm(amp, phase, psi, tol=1e-5) -> dict:
    amp_norm = np.sum(amp**2, axis=-1)
    phase_abs = np.abs(phase)
    psi_norm = np.sum(np.abs(psi)**2, axis=-1)
    if not np.allclose(amp_norm, 1.0, atol=tol):
        raise ValueError("Amplitude vectors are not normalized to unit norm.")
    if not np.allclose(phase_abs, 1.0, atol=tol):
        raise ValueError("Phase factors do not have unit magnitude.")
    if not np.allclose(psi_norm, 1.0, atol=tol):
        raise ValueError("Resulting quantum states are not normalized to unit norm.")

    logger.debug("Quantum state normalization check passed.")
    logger.debug(f'Amplitude norm: min={amp_norm.min():.5f}, max={amp_norm.max():.5f}')
    logger.debug(f'Phase magnitude: min={phase_abs.min():.5f}, max={phase_abs.max():.5f}')
    logger.debug(f'Quantum state norm: min={psi_norm.min():.5f}, max={psi_norm.max():.5f}')


def __validate_weights(weights):
    if weights is None:
        return set()
    
    if isinstance(weights, str):
        weights = {weights}
    if isinstance(weights, (list, set, tuple)):
        weights = set(weights)
    if weights - {'amp', 'phase'}:
        raise ValueError(f"Invalid weights specified: {weights}. Allowed values are 'amp' and 'phase'.")
    return weights




def plot(
    x: np.ndarray,
    i: int,
    encoding: str | None = None,
    file_name: str | None = None,
    title: str | None = None,
    figsize: list[float, float] | tuple[float, float] = (4.0, 4.0),
    dpi: int = 300,
    xlim: float | None = None,
    ylim: float | None = None,
    alpha: float = 1.0,
) -> None:
    """
    Plot state vector in the complex plane.

    Parameters
    ----------
    x : numpy.ndarray, shape (n_genes, n_components) or (n_genes, n_replicates, n_components)
        State-vector array returned by ``build``.
        If 3D, replicates are averaged before plotting.

    i : int
        Row index of the gene to plot.

    encoding : str or None, optional
        Encoding used to build ``x``. Set this to an encoding containing
        ``"OD"``, such as ``"ODTDP"`` or ``"ODIDP"``, to display the two
        orthogonalized direction branches separately.

    file_name : str or None, optional
        Output file path. If provided, the figure is saved to this path.
        If ``None``, the figure is displayed.

    title : str or None, optional
        Plot title.

    figsize : tuple of float, default=(4.0, 4.0)
        Figure size in inches.

    dpi : int, default=300
        Figure resolution in dots per inch.

    xlim : tuple of float or None, optional
        Limits of the real axis. If ``None``, limits are determined
        automatically.

    ylim : tuple of float or None, optional
        Limits of the imaginary axis. If ``None``, the same limits as
        ``xlim`` are used.

    alpha : float, default=1.0
        Transparency of plotted points and arrows.
    """
    if x.ndim == 3:
        state_vec = np.nanmean(x[i, :, :], axis=0)
    elif x.ndim == 2:
        state_vec = x[i, :]
    else:
        raise ValueError("x must be a 2D or 3D array.")
    
    if (encoding is None) or ("OD" not in encoding):
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        __plot_qstate(ax, state_vec, False, title, xlim, ylim, alpha)
        
    else:
        if (len(state_vec) % 2) != 0:
            raise ValueError(f"State vector length must be even for orthogonalized encodings, but got length {len(state_vec)}.")
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(1, 1, 1)
        __plot_qstate(ax, state_vec[0::2], False, title, xlim, ylim, alpha)
        __plot_qstate(ax, state_vec[1::2], True, title, xlim, ylim, alpha)

    utils._close_plt(file_name)


def __plot_qstate(ax, state_vec, substyle, title, xlim, ylim, alpha, eps=1e-12):
    if substyle:
        arrowstyle = '->'
        linestyle = '--'
    else:
        arrowstyle = '-|>'
        linestyle = '-'
    
    colors = [utils._COLORS[i % len(utils._COLORS)] for i in range(len(state_vec))]
    for j in range(len(state_vec)):
        if np.abs(state_vec[j].real) > eps:
            ax.annotate("", xy=(state_vec[j].real, state_vec[j].imag), xytext=(0, 0),
                arrowprops=dict(arrowstyle=arrowstyle, linestyle=linestyle,
                                facecolor=colors[j], alpha=alpha, fc=colors[j], ec=colors[j],
                                shrinkA=0, shrinkB=0))
        ax.scatter(state_vec[j].real, state_vec[j].imag, color=colors[j], alpha=alpha, s=5)

    if xlim is None:
        xlim = np.sqrt((state_vec.real**2 + state_vec.imag**2)).max() * 1.1
        xlim = (-xlim, xlim)
    if ylim is None:
        ylim = xlim

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if title is not None:
        ax.set_title(title)
    ax.grid(True)
    ax.autoscale(False)

    
        



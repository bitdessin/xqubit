from __future__ import annotations
import tqdm
from typing import TYPE_CHECKING
import logging
import numpy as np

logging.basicConfig()
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator


__not_implemented_msg = "Metric computation is not supported for data retaining replicate structure."


def swaptest(
    x: np.ndarray,
    n: int | None = None,
    backend: AerSimulator | None = None,
    shots: int = 2**13,
    execute: bool = True,
    seed: int | None = None
) -> np.ndarray:
    """
    Estimate pairwise fidelity using SWAP test circuits.

    This function estimates gene-gene fidelity values from normalized state
    vectors by running a SWAP test for each pair of genes.

    For two normalized state vectors, the SWAP test estimates the probability
    of measuring the ancillary qubit as 0. The fidelity is then calculated from
    that probability. The result can be used as a gene-gene similarity matrix.

    Parameters
    ----------
    x : numpy.ndarray, shape (n_genes, n_components)
        Normalized state vectors. Each row corresponds to one gene.

    n : int or None, optional
        Number of gene pairs to sample. If ``None``, all upper-triangular gene
        pairs are evaluated. If an integer is given, only that many pairs are
        randomly selected.

    backend : AerSimulator or None, optional
        Qiskit backend used to run the SWAP test circuits. If ``None``, a
        default ``AerSimulator`` backend is created.

    shots : int, default=2**13
        Number of measurement shots used for each SWAP test circuit.

    execute : bool, default=True
        If ``True``, run the SWAP test circuits and return a fidelity matrix.

        If ``False``, return a representative SWAP test circuit constructed
        from the first two rows of ``x``. This is useful for inspecting or
        drawing the circuit.

    seed : int or None, optional
        Random seed used when ``n`` is specified.

    Returns
    -------
    numpy.ndarray or QuantumCircuit
        If ``execute=True``, returns a symmetric matrix of pairwise fidelity
        estimates with shape (n_genes, n_genes).

        If ``execute=False``, returns a Qiskit ``QuantumCircuit`` object for
        the first two state vectors.

    Notes
    -----
    The returned matrix contains ``NaN`` for pairs that are not evaluated when
    ``n`` is specified.
    """    
    from qiskit import transpile
    rng = np.random.default_rng(seed)

    if x.ndim != 2:
        raise NotImplementedError(__not_implemented_msg)

    if shots <= 0:
        raise ValueError("shots must be positive.")
            
    if backend is None:
        from qiskit_aer import AerSimulator
        backend = AerSimulator(method="statevector")
    

    def __swap_test(qc, shots, backend):
        qc = transpile(qc, backend, optimization_level=1)
        result = backend.run(qc, shots=shots).result()
        counts = result.get_counts()
        p0 = counts.get("0", 0) / shots
        return max(0.0, 2.0 * p0 - 1.0)

    if execute is False:
        return __build_swaptest_circuit(x[0], x[1])

    # fidelity computation for all gene pairs
    fmat = np.full((x.shape[0], x.shape[0]), np.nan)
    gi, gj = np.triu_indices(x.shape[0])

    if n is not None:
        idx = rng.choice(len(gi), size=min(n, len(gi)), replace=False)
        gi, gj = gi[idx], gj[idx]

    for i, j in tqdm.tqdm(zip(gi, gj), total=len(gi), desc="swap test", leave=False):
        qc = __build_swaptest_circuit(x[i], x[j])
        fmat[i, j] = fmat[j, i] = __swap_test(qc, shots, backend)

    return fmat


def __build_swaptest_circuit(x: np.ndarray, y: np.ndarray) -> QuantumCircuit:
    from qiskit import QuantumCircuit

    def __pad_qstate(psi: np.ndarray) -> np.ndarray:
        if len(psi) == 0:
            raise ValueError("Quantum state vector cannot be empty.")
        
        n = int(2 ** np.ceil(np.log2(len(psi))))
        if n > len(psi):
            psi = np.pad(psi, (0, n - len(psi)))
        
        norm = np.linalg.norm(psi)
        psi = psi if norm == 0 else psi / norm
        return psi

    def __sort_qstate(psi):
        n = int(np.log2(len(psi)))
        psi = psi.reshape([2]*n)
        psi = np.transpose(psi, axes=list(reversed(range(n))))
        return psi.flatten()

    x = __sort_qstate(__pad_qstate(x))
    y = __sort_qstate(__pad_qstate(y))
    n_qubits = int(np.log2(len(x)))

    anc = 0
    regx = list(range(1, 1 + n_qubits))
    regy = list(range(1 + n_qubits, 1 + 2 * n_qubits))

    qc = QuantumCircuit(1 + 2 * n_qubits, 1)
    qc.initialize(x, regx)
    qc.initialize(y, regy)
    qc.h(anc)
    for q1, q2 in zip(regx, regy):
        qc.cswap(anc, q1, q2)
    qc.h(anc)
    qc.measure(anc, 0)

    return qc

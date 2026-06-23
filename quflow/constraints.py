"""
Simple coastline boundary conditions for the quantized Poisson equation.

This is a small version of ``constraintsTest.py`` with one fixed numerical path:

1. The boundary condition is the default commutator condition

       [F_c, P] = F_c P - P F_c = 0.

2. Independent rows of this condition are selected by pivoted QR.

3. The constrained Poisson equation is solved by the Schur complement.

The unknown matrix P is stored as the vector ``p = P.ravel()``.  With this
notation the linear system is

       A p + V^H lambda = w,
       V p              = 0,

where A is the Poisson matrix and V contains the independent boundary
condition rows.  Eliminating p gives a smaller Schur-complement system for
lambda.  The expensive factorizations are done once in ``__init__``; each
call to ``solve`` then uses those factorizations.
"""

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import quflow as qf


__all__ = [
    "BoundaryConditionPoisson",
    "CoastlinePoisson",
    "commutator_matrix",
    "rank_from_eigenvalues",
    "commutator_rank",
    "complex_matrix_to_real_vector",
    "complex_vector_to_real_vector",
    "linear_operator",
    "make_trace_free_on_U",
    "plot_eigenvalues",
    "plot_with_countour",
    "project",
    "project_soft",
    "quantize_to_three_levels",
    "real_vector_to_complex_matrix",
    "real_vector_to_complex_vector",
    "select_independent_rows_by_qr",
    "scipy",
    "level_set_eigenvects",
    "island_mask",
]


def commutator_matrix(F):
    """
    Return the sparse matrix C such that

        C @ P.ravel() == (F @ P - P @ F).ravel().

    Here F and P are N by N matrices.  The matrix C has size N**2 by N**2.
    """
    N = F.shape[0]
    identity = sp.eye(N, format="csr", dtype=F.dtype)
    left_multiplication = sp.kron(F, identity, format="csr")
    right_multiplication = sp.kron(identity, F.T, format="csr")
    return left_multiplication - right_multiplication


def _eigenvalue_multiplicities(eigenvalues, relative_tolerance=1e-10):
    """
    Group nearly equal eigenvalues and return the group sizes.

    If F has eigenvalue groups of sizes n_1, ..., n_k, then the matrices
    commuting with F have dimension n_1**2 + ... + n_k**2.
    """
    eigenvalues = np.sort(np.asarray(eigenvalues, dtype=float))
    if eigenvalues.size == 0:
        return []

    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    tolerance = relative_tolerance * scale

    counts = []
    group_start = eigenvalues[0]
    group_size = 1
    for value in eigenvalues[1:]:
        if abs(value - group_start) <= tolerance:
            group_size += 1
        else:
            counts.append(group_size)
            group_start = value
            group_size = 1
    counts.append(group_size)
    return counts


def commutator_rank(F, relative_tolerance=1e-10):
    """
    Return the rank of the map P -> [F, P].

    The rank is computed from the eigenvalue multiplicities of F:

        rank([F, .]) = N**2 - sum(multiplicity**2).
    """
    N = F.shape[0]
    # F is skew-Hermitian in this code.  Therefore 1j*F is Hermitian and has
    # real eigenvalues, which are safe to group by numerical equality.
    eigenvalues = np.linalg.eigvalsh(1j * F).real
    multiplicities = _eigenvalue_multiplicities(
        eigenvalues, relative_tolerance=relative_tolerance
    )
    rank = N**2 - sum(size**2 for size in multiplicities)
    return int(rank), multiplicities


def select_independent_rows_by_qr(C, rank):
    """
    Select a sparse, well-scaled row basis for C p = 0.

    The steps are:

    1. Remove zero rows and normalize every remaining row to unit length.
    2. Sort the rows from least dense to most dense.
    3. Use column-pivoted QR on the transposed row matrix to choose a basis.

    We keep actual rows of C, so the result stays sparse.
    """
    C = C.tocsr()
    if rank == 0:
        return sp.csr_matrix((0, C.shape[1]), dtype=C.dtype)

    row_norms = np.sqrt(np.array(C.multiply(C.conj()).sum(axis=1)).real.ravel())
    nonzero_rows = np.flatnonzero(row_norms > 0)
    if nonzero_rows.size < rank:
        raise ValueError(
            f"Only {nonzero_rows.size} nonzero constraint rows are available, "
            f"but the expected rank is {rank}."
        )

    row_densities = C.getnnz(axis=1)
    sparsity_order = np.lexsort((nonzero_rows, row_densities[nonzero_rows]))
    sorted_rows = nonzero_rows[sparsity_order]

    normalized_rows = C[sorted_rows, :].copy()
    normalized_rows = normalized_rows.multiply(
        (1.0 / row_norms[sorted_rows])[:, np.newaxis]
    ).tocsr()

    _, _, pivots = scipy.linalg.qr(
        normalized_rows.toarray().T, pivoting=True, mode="economic"
    )
    return normalized_rows[pivots[:rank], :].tocsr()


def level_set_eigenvects(F, center, num_of_eigenvects, linear=False):
    """
    Keep the eigenvectors of F whose eigenvalues are closest to one level.

    Since F is skew-Hermitian, -1j*F is Hermitian with real eigenvalues.  We
    keep the ``num_of_eigenvects`` eigenvalues closest to ``center`` and set
    all other eigenvalues to zero.

    Returns
    -------
    F_c : ndarray
        The truncated coastline matrix.
    K : int
        The number of kept eigenvectors.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(-1j * F)
    K = int(num_of_eigenvects)
    if K < 0 or K > eigenvalues.size:
        raise ValueError(
            f"num_of_eigenvects must be between 0 and {eigenvalues.size}, got {K}."
        )

    keep = np.zeros(eigenvalues.shape, dtype=bool)
    closest = np.argsort(np.abs(eigenvalues - center), kind="stable")[:K]
    keep[closest] = True

    kept_eigenvalues = np.zeros_like(eigenvalues)
    if linear:
        kept_eigenvalues[keep] = np.arange(1, K + 1)
    else:
        kept_eigenvalues[keep] = eigenvalues[keep]

    F_c = (eigenvectors * (1j * kept_eigenvalues)[np.newaxis, :]) @ (
        eigenvectors.conj().T
    )
    return F_c, K


class BoundaryConditionPoisson:
    """
    Solve -Delta P = W with the boundary condition [F_c, P] = 0.

    The setup work is done once:

    1. Build C for [F_c, P] = 0.
    2. Pick independent rows V of C by QR.
    3. Factor the Poisson matrix with boundary conditions.
    4. Form and factor the Schur complement.
    """

    def __init__(self, F_c, N=None):
        if N is None:
            N = F_c.shape[0]
        if F_c.shape != (N, N):
            raise ValueError(f"F_c must have shape {(N, N)}, got {F_c.shape}.")

        self.N = N
        self.constraint_rank, self.eigenvalue_group_sizes = commutator_rank(F_c)

        C = commutator_matrix(F_c)
        self.V = select_independent_rows_by_qr(C, self.constraint_rank)
        self.n_constraints = self.V.shape[0]

        self._factor_poisson_matrix()
        self._factor_schur_complement()

    def _factor_poisson_matrix(self):
        """
        Factor A with boundary conditions.

        This version of A is nonsingular, which lets us use the Schur
        complement directly.  The trace of P is removed after each solve.
        """
        A = qf.laplacian.sparse.laplacian(self.N, bc=True)
        self._A_lu = spla.splu(A.tocsc())

    def _factor_schur_complement(self):
        """
        Form and factor -V A^{-1} V^H.

        This is the expensive setup step.  It costs one Poisson solve per
        constraint row.  After this, each call to solve needs only two Poisson
        solves and one small dense solve.
        """
        m = self.n_constraints
        if m == 0:
            self._schur_lu = None
            return

        V_H = self.V.conj().T.tocsc()
        negative_schur = np.empty((m, m), dtype=complex)

        for j in range(m):
            right_hand_side = V_H[:, j].toarray().ravel()
            A_inverse_column = self._A_lu.solve(right_hand_side)
            negative_schur[:, j] = -(self.V @ A_inverse_column)

        self._schur_lu = scipy.linalg.lu_factor(negative_schur)

    def solve(self, W):
        """
        Solve for P.

        W may be one N by N matrix, or an array whose last two dimensions are
        N by N.  The returned array has the same shape as W.
        """
        W = np.asarray(W)
        if W.shape[-2:] != (self.N, self.N):
            raise ValueError(
                f"W must end with shape {(self.N, self.N)}, got {W.shape}."
            )

        if W.ndim == 2:
            return self._solve_one(W)

        batch_shape = W.shape[:-2]
        matrices = W.reshape((-1, self.N, self.N))
        solved = [self._solve_one(matrix) for matrix in matrices]
        return np.asarray(solved).reshape(batch_shape + (self.N, self.N))

    def _solve_one(self, W):
        """Solve one constrained Poisson problem."""
        w = W.ravel()
        p_without_constraints = self._A_lu.solve(w)

        if self.n_constraints:
            schur_right_hand_side = self.V @ p_without_constraints
            lambda_vector = scipy.linalg.lu_solve(
                self._schur_lu, -schur_right_hand_side
            )

            correction = self.V.conj().T @ lambda_vector
            p = self._A_lu.solve(w - correction)
        else:
            p = p_without_constraints

        P = p.reshape(self.N, self.N)
        P -= np.trace(P) / self.N * np.eye(self.N, dtype=P.dtype)
        return P

    def solve_poisson(self, W):
        """Alias with the same meaning as ``solve``."""
        return self.solve(W)

    def boundary_residual(self, P):
        """Return || V @ P.ravel() ||, useful for checking a solution."""
        return float(np.linalg.norm(self.V @ np.asarray(P).ravel()))


def island_mask(N, center, radius=1, p=2):
    "Build mask for island w.r.t. p-norm."
    theta, phi = np.meshgrid(np.linspace(0,np.pi, N), \
                             np.linspace(0,2*np.pi, 2*N-1, endpoint=False), indexing='ij')
    
    
    center = np.asarray(center)
    center /= np.linalg.norm(center)

    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)

    return np.abs(x-center[0])**p + np.abs(y-center[1])**p + np.abs(z-center[2])**p < radius**p 


# The constraint-cluster notebook uses the fuller experimental solver that
# lived in constraintsTest.py.  Re-export those notebook-facing functions here
# so user code can depend on the stable quflow.constraints module path.
from .constraintsTest import (  # noqa: E402
    complex_matrix_to_real_vector,
    complex_vector_to_real_vector,
    linear_operator,
    make_trace_free_on_U,
    plot_eigenvalues,
    plot_with_countour,
    project,
    project_soft,
    quantize_to_three_levels,
    rank_from_eigenvalues,
    real_vector_to_complex_matrix,
    real_vector_to_complex_vector,
)
from .constraintsTest import CoastlinePoisson as _NotebookCoastlinePoisson  # noqa: E402


class CoastlinePoisson(BoundaryConditionPoisson):
    """
    Full coastline-constrained Poisson solver used by the cluster notebook.

    In addition to the methods implemented in ``constraintsTest.py``
    (``"lu"``, ``"qr"``, and ``"svd"``), this wrapper accepts the method names
    found in older notebook cells:

    ``"im-matrix"``
        Use the default commutator matrix with sparse row selection.
    ``"im"``
        Use per-eigenvector projector constraints with sparse row selection.
    """

    def __init__(self, F_c, N=None, method=None, solver=None, **kwargs):
        if method is None and solver is None and not kwargs:
            super().__init__(F_c, N=N)
            self._notebook_solver = None
            return

        method = "lu" if method is None else method
        solver = "direct" if solver is None else solver
        each_eigenvector = kwargs.pop("each_eigenvector", False)
        if method == "im-matrix":
            method = "lu"
        elif method == "im":
            method = "lu"
            each_eigenvector = True

        self._notebook_solver = _NotebookCoastlinePoisson(
            F_c,
            N=N,
            method=method,
            solver=solver,
            each_eigenvector=each_eigenvector,
            **kwargs,
        )

    def __getattr__(self, name):
        notebook_solver = self.__dict__.get("_notebook_solver")
        if notebook_solver is not None:
            return getattr(notebook_solver, name)
        raise AttributeError(name)

    def solve(self, W):
        if self._notebook_solver is not None:
            return self._notebook_solver.solve(W)
        return super().solve(W)

    def solve_poisson(self, W):
        return self.solve(W)

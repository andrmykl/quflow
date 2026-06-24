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
import scipy.linalg.interpolative
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


def _commutator_linear_operator(F):
    """Return matrix-free operator for P -> F @ P - P @ F."""
    N = F.shape[0]
    n2 = N * N

    def matvec(x):
        X = x.reshape((N, N))
        return (F @ X - X @ F).ravel()

    def rmatvec(x):
        X = x.reshape((N, N))
        return (F.conj().T @ X - X @ F.conj().T).ravel()

    return spla.LinearOperator(
        shape=(n2, n2),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=F.dtype,
    )


def _commutator_row_norms(F):
    """Return Euclidean norms of rows of the commutator matrix."""
    row_sq = np.sum(np.abs(F) ** 2, axis=1)
    col_sq = np.sum(np.abs(F) ** 2, axis=0)
    diag = np.diag(F)
    overlap = (
        -np.abs(diag)[:, np.newaxis] ** 2
        -np.abs(diag)[np.newaxis, :] ** 2
        + np.abs(diag[:, np.newaxis] - diag[np.newaxis, :]) ** 2
    )
    norms_sq = row_sq[:, np.newaxis] + col_sq[np.newaxis, :] + overlap
    return np.sqrt(np.maximum(norms_sq.real, 0.0)).ravel()


def _commutator_row_densities(F):
    """Return approximate real row densities of the complex commutator rows."""
    support = np.abs(F) > 0
    row_counts = np.sum(support, axis=1)
    col_counts = np.sum(support, axis=0)
    diag_support = np.diag(support)
    duplicate = diag_support[:, np.newaxis] & diag_support[np.newaxis, :]
    densities = row_counts[:, np.newaxis] + col_counts[np.newaxis, :] - duplicate
    return (2 * densities.astype(int)).ravel()


def _selected_commutator_rows(F, selected):
    """Build selected sparse rows of the commutator matrix."""
    N = F.shape[0]
    n2 = N * N
    selected = np.asarray(selected, dtype=int)
    rows = []
    cols = []
    vals = []

    for row_pos, idx in enumerate(selected):
        a = idx // N
        b = idx % N

        row_a = F[a, :]
        nz_row = np.flatnonzero(np.abs(row_a) > 0)
        rows.extend([row_pos] * nz_row.size)
        cols.extend((nz_row * N + b).tolist())
        vals.extend(row_a[nz_row].tolist())

        col_b = F[:, b]
        nz_col = np.flatnonzero(np.abs(col_b) > 0)
        rows.extend([row_pos] * nz_col.size)
        cols.extend((a * N + nz_col).tolist())
        vals.extend((-col_b[nz_col]).tolist())

    V = sp.coo_matrix((vals, (rows, cols)), shape=(selected.size, n2)).tocsr()
    V.sum_duplicates()
    V.eliminate_zeros()
    return V


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


def select_independent_rows_by_qr(
    C,
    rank,
    selection="qr",
    normalize_rows=True,
    verbose=False,
    row_norms=None,
    row_densities=None,
    row_norm_tol=1e-12,
    get_rows=None,
):
    """
    Select a sparse, well-scaled row basis for C p = 0.

    The steps are:

    1. Remove zero rows and optionally normalize every remaining row to unit
       length.
    2. Sort the rows from least dense to most dense when row densities are
       available.
    3. Use column-pivoted QR or interpolative decomposition on the
       transposed row matrix to choose a basis.

    We keep actual rows of C, so the result stays sparse.
    """
    matrix_free = isinstance(C, spla.LinearOperator)
    if rank == 0:
        return sp.csr_matrix((0, C.shape[1]), dtype=C.dtype)

    if matrix_free:
        if selection != "interpolative":
            raise ValueError("Matrix-free row selection requires 'interpolative'.")
        if row_norms is None or get_rows is None:
            raise ValueError(
                "Matrix-free row selection requires row_norms and get_rows."
            )
        row_norms = np.asarray(row_norms)
        if row_densities is not None:
            row_densities = np.asarray(row_densities)
    else:
        C = C.tocsr()
        row_norms = np.sqrt(
            np.array(C.multiply(C.conj()).sum(axis=1)).real.ravel()
        )
        row_densities = C.getnnz(axis=1)

    if row_norm_tol < 0:
        raise ValueError(f"row_norm_tol must be nonnegative, got {row_norm_tol}.")

    max_row_norm = float(np.max(row_norms)) if row_norms.size else 0.0
    row_norm_cutoff = row_norm_tol * max_row_norm
    nonzero_rows = np.flatnonzero(row_norms > row_norm_cutoff)
    if nonzero_rows.size < rank:
        raise ValueError(
            f"Only {nonzero_rows.size} constraint rows have norm above "
            f"{row_norm_cutoff:.3e}, but the expected rank is {rank}."
        )

    if matrix_free:
        if row_densities is None:
            sorted_rows = nonzero_rows
        else:
            sparsity_order = np.lexsort((nonzero_rows, row_densities[nonzero_rows]))
            sorted_rows = nonzero_rows[sparsity_order]
        if normalize_rows:
            sorted_scales = row_norms[sorted_rows]
        else:
            sorted_scales = np.ones(sorted_rows.size, dtype=row_norms.dtype)

        def matvec(x):
            weighted = np.zeros(C.shape[0], dtype=C.dtype)
            weighted[sorted_rows] = x / sorted_scales
            return C.rmatvec(weighted)

        def rmatvec(x):
            return C.matvec(x)[sorted_rows] / sorted_scales

        transposed_rows = spla.LinearOperator(
            shape=(C.shape[1], sorted_rows.size),
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=C.dtype,
        )
    else:
        sparsity_order = np.lexsort((nonzero_rows, row_densities[nonzero_rows]))
        sorted_rows = nonzero_rows[sparsity_order]

        candidate_rows = C[sorted_rows, :].copy().tocsr()
        if normalize_rows:
            candidate_rows = candidate_rows.multiply(
                (1.0 / row_norms[sorted_rows])[:, np.newaxis]
            ).tocsr()

    if selection == "qr":
        _, pivots = scipy.linalg.qr(
            candidate_rows.toarray().T, pivoting=True, mode="r"
        )
    elif selection == "interpolative":
        if not matrix_free:
            transposed_rows = candidate_rows.conj().transpose().toarray()
        pivots, _ = scipy.linalg.interpolative.interp_decomp(
            transposed_rows, rank, rand=False
        )
    else:
        raise ValueError(
            f"Unknown row selection '{selection}'. "
            "Use 'qr' or 'interpolative'."
        )

    if matrix_free:
        selected_rows = sorted_rows[np.asarray(pivots[:rank], dtype=int)]
        V = get_rows(selected_rows)
        if normalize_rows:
            V = V.multiply((1.0 / row_norms[selected_rows])[:, np.newaxis])
        V = V.tocsr()
    else:
        V = candidate_rows[pivots[:rank], :].tocsr()
    if verbose:
        dropped_rows = np.count_nonzero((row_norms > 0) & (row_norms <= row_norm_cutoff))
        density = V.nnz / max(V.shape[0] * V.shape[1], 1)
        sparsity = 1.0 - density
        print(
            f"Selected constraint matrix V: shape={V.shape}, "
            f"nnz={V.nnz}, density={density:.6e}, "
            f"sparsity={sparsity:.6e}, normalized={normalize_rows}, "
            f"row_norm_tol={row_norm_tol:.1e}, dropped_tiny_rows={dropped_rows}"
        )
    return V


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

    def __init__(
        self,
        F_c,
        N=None,
        row_selection="qr",
        normalize_rows=True,
        row_norm_tol=1e-12,
        verbose=False,
    ):
        if N is None:
            N = F_c.shape[0]
        if F_c.shape != (N, N):
            raise ValueError(f"F_c must have shape {(N, N)}, got {F_c.shape}.")

        self.N = N
        self.constraint_rank, self.eigenvalue_group_sizes = commutator_rank(F_c)

        if row_selection == "interpolative_operator":
            C = _commutator_linear_operator(F_c)
            selection = "interpolative"
            row_norms = _commutator_row_norms(F_c)
            row_densities = _commutator_row_densities(F_c)
            get_rows = lambda rows: _selected_commutator_rows(F_c, rows)
        elif row_selection in {"qr", "interpolative"}:
            C = commutator_matrix(F_c)
            selection = row_selection
            row_norms = None
            row_densities = None
            get_rows = None
        else:
            raise ValueError(
                f"Unknown row selection '{row_selection}'. "
                "Use 'qr', 'interpolative', or 'interpolative_operator'."
            )

        self.V = select_independent_rows_by_qr(
            C,
            self.constraint_rank,
            selection=selection,
            normalize_rows=normalize_rows,
            verbose=verbose,
            row_norms=row_norms,
            row_densities=row_densities,
            row_norm_tol=row_norm_tol,
            get_rows=get_rows,
        )
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
        if method is None and solver is None and set(kwargs) <= {
            "row_selection",
            "normalize_rows",
            "row_norm_tol",
            "verbose",
        }:
            super().__init__(F_c, N=N, **kwargs)
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

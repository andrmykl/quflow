"""
Coastline-constrained Poisson solver for Zeitlin's model.

Solves  -Delta_N P = W  subject to  [F, P] = 0,
where F is a quantized indicator function whose level sets
define the coastline.

Two constraint-extraction methods are available:
  - "lu"   : selects independent rows of the sparse commutator
             matrix C via QR-pivoted row selection (fast, stays sparse).
  - "qr"   : selects independent sparse rows after row equilibration
             and broader candidate sampling. Usually better conditioned
             than "lu" while remaining sparse.
  - "qr-simple" : applies pivoted QR directly to the transpose of the
             nonzero rows and returns the selected sparse rows.
  - "im"   : uses an interpolative decomposition of the commutator
             operator itself (matrix-free from F_c) to select a sparse
             skeleton row basis.
  - "im-matrix" : applies interpolative decomposition to the explicitly
             constructed row-equilibrated matrix C^H with ``rand=False``.
  - "qd"   : uses a Q-DEIM selection on an orthonormal basis of the
             row space to choose sparse constraint rows.
  - "svd"  : computes a truncated SVD of C to obtain a dense but
             well-conditioned constraint basis (slower, dense block).

Constraint-matrix construction options:
  - default : build C from the single truncated matrix F_c via [F_c, P] = 0.
  - each_eigenvector=True : build C from the rank-1 projector constraints
             [e_i e_i^H, P] = 0 for every eigenvector e_i whose eigenvalue
             in F_c is nonzero. This tracks the per-eigenvector condition
             directly and is typically denser.

Four solver strategies are available:
  - "direct"   : sparse LU factorization of the full KKT saddle-point
                  matrix.  Pairs naturally with method="lu" (sparse V
                  keeps fill-in low).
  - "schur"    : explicit Schur complement — forms and LU-factors the
                  dense m x m matrix S = V A_bc^{-1} V^H.  2 Poisson
                  solves per step; m Poisson solves at setup.
  - "schur-poisson" : same Schur complement setup as "schur", but uses
                  ``qf.laplacian.solve_poisson(..., bc=True)`` in the
                  repeated solve phase.
  - "schur_cg" : CG on the Schur complement with a symmetric
                  regularization of A.  Preconditioned + warm-started.
  - "minres"   : MINRES on the **same** full KKT as ``direct`` (no
                  ``A_reg``), real 2(N²+m+1) unknowns.  Sparse matvecs
                  only; default maxiter is large (saddle-point + tight
                  ``rtol`` is expensive).  Warm-started.

Typical usage
-------------
>>> from coastline import CoastlinePoisson
>>> solver = CoastlinePoisson(F_c, N, method="lu", solver="direct")
>>> P = solver.solve(W)
>>> # explicit Schur complement (fast per-step, expensive setup):
>>> solver = CoastlinePoisson(F_c, N, method="svd", solver="schur")
>>> P = solver.solve(W)
>>> # iterative Schur complement (CG):
>>> solver = CoastlinePoisson(F_c, N, method="lu", solver="schur_cg")
>>> P = solver.solve(W)
>>> # MINRES on the full KKT system (no factorization):
>>> solver = CoastlinePoisson(F_c, N, method="lu", solver="minres")
>>> P = solver.solve(W)
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg
import scipy.linalg.interpolative
import quflow as qf
from scipy.linalg import solve_continuous_lyapunov


# ---------------------
# Vectorized operators
# ---------------------

def _kron_right(A, N):
    """Return sparse matrix implementing vecc(A @ B) from vecc(B)."""
    return sp.kron(A, sp.eye(N, format="csr"), format="csr")


def _kron_left(A, N):
    """Return sparse matrix implementing vecc(B @ A) from vecc(B)."""
    return sp.kron(sp.eye(N, format="csr"), A.T, format="csr")


def commutator_matrix(F, N=None):
    """
    Build sparse matrix C such that C @ vecc(P) = vecc([F, P]).

    Parameters
    ----------
    F : ndarray, shape (N, N)
    N : int, optional

    Returns
    -------
    C : sparse matrix, shape (N^2, N^2)
    """
    if N is None:
        N = F.shape[0]
    return _kron_right(F, N) - _kron_left(F, N)


def rank_from_eigenvalues(F, tol=1e-10):
    """
    Compute the rank of the commutator-with-F operator from the
    eigenvalues of F alone.

    Returns
    -------
    rank : int
        Rank of the commutator operator [F, .].
    K : int
        Number of eigenvalue groups with more than one member
        that are degenerate (for diagnostics).
    multiplicities : list of int
        Multiplicity of each distinct eigenvalue.
    """
    eigvals = np.linalg.eigvalsh(1j * F).real
    rounded = np.round(eigvals / max(tol, np.max(np.abs(eigvals)) * tol)) * max(tol, np.max(np.abs(eigvals)) * tol)
    _, counts = np.unique(np.round(rounded, decimals=8), return_counts=True)
    N = F.shape[0]
    rank = N**2 - int(np.sum(counts**2))
    return rank, counts.tolist()


def truncate_to_level_set(F, center, epsilon, linear=False):
    """
    Keep only the eigenvectors of F whose eigenvalues lie in the band
    [center - epsilon, center + epsilon], and zero out the rest.

    If linear=False, retain the original eigenvalues inside the band.

    If linear=True, replace the retained eigenvalues by distinct nonzero
    values 1j * 1, 1j * 2, ..., 1j * K, so that none of the kept modes
    merges with the discarded zero eigenspace.

    Parameters
    ----------
    F : ndarray, shape (N, N)
        Skew-Hermitian matrix.
    center : float
        Target level-set value (imaginary part).
    epsilon : float
        Half-width of the retained band.
    linear : bool, default False
        Whether to remap the retained eigenvalues to distinct nonzero values.

    Returns
    -------
    F_c : ndarray, shape (N, N)
        Truncated matrix.
    K : int
        Number of retained eigenvalues.
    """
    # Since F is skew-Hermitian, -1j*F is Hermitian with real eigenvalues.
    eigvals, eigvecs = np.linalg.eigh(-1j * F)

    mask = (center - epsilon <= eigvals) & (eigvals <= center + epsilon)
    K = int(np.sum(mask))

    eigvals_out = np.zeros_like(eigvals)
    if linear:
        eigvals_out[mask] = np.arange(1, K + 1, dtype=eigvals.dtype)
    else:
        eigvals_out[mask] = eigvals[mask]

    F_c = eigvecs @ (1j * np.diag(eigvals_out)) @ eigvecs.conj().T
    return F_c, K



def quantize_to_three_levels(F, targets=(-1.0, 0.0, 1.0), tol=1e-10):
    """
    Remap the eigenvalues of a skew-Hermitian F so that F_c has exactly
    three distinct purely imaginary eigenvalue levels.

    F is assumed to have exactly three eigenvalues with nonzero imaginary
    part.  These are sorted by imaginary part and replaced with
    ``1j * targets[0]``, ``1j * targets[1]``, ``1j * targets[2]``
    (ascending order).  All remaining eigenvalues (those with
    |Im(λ)| < tol) are set to ``1j * targets[1]`` (the middle level),
    so that the middle coastline eigenvalue merges with the bulk to
    form a single degenerate group.

    The result is a skew-Hermitian F_c with exactly three distinct
    eigenvalues and well-controlled eigenvalue gaps, ensuring the
    discrete gradient of F_c normal to the coastline is O(1).

    Parameters
    ----------
    F : ndarray, shape (N, N)
        Skew-Hermitian matrix with exactly three eigenvalues whose
        imaginary part is significantly nonzero.
    targets : tuple of three floats, default (-1.0, 0.0, 1.0)
        The three target imaginary eigenvalue levels in ascending order.
        The eigenvalues of F_c will be ``1j * targets[k]``.
    tol : float
        Threshold for deciding whether an eigenvalue's imaginary part
        is "nonzero".

    Returns
    -------
    F_c : ndarray, shape (N, N)
        Skew-Hermitian matrix with exactly three distinct eigenvalues.
    multiplicities : tuple of int
        (n_minus, n_zero, n_plus) — the multiplicity of each level.
    """
    t_lo, t_mid, t_hi = sorted(targets)

    eigvals, eigvecs = np.linalg.eigh(1j * F)
    # eigvals are now real (eigenvalues of the Hermitian matrix 1j*F).
    # Original skew-Hermitian eigenvalues were 1j * (-eigvals).
    imag_parts = -eigvals  # imaginary parts of the original eigenvalues

    nonzero_mask = np.abs(imag_parts) > tol
    n_nonzero = int(np.sum(nonzero_mask))
    if n_nonzero != 3:
        raise ValueError(
            f"Expected exactly 3 eigenvalues with |Im(λ)| > {tol}, "
            f"got {n_nonzero}."
        )

    nonzero_indices = np.where(nonzero_mask)[0]
    nonzero_imag = imag_parts[nonzero_indices]
    order = np.argsort(nonzero_imag)

    new_imag = np.full_like(imag_parts, t_mid)
    new_imag[nonzero_indices[order[0]]] = t_lo
    new_imag[nonzero_indices[order[1]]] = t_mid
    new_imag[nonzero_indices[order[2]]] = t_hi

    # Reconstruct: F_c = eigvecs @ diag(-1j * new_imag) @ eigvecs^H
    # Since eigvecs come from eigh(1j*F), F_c = -1j * eigvecs @ diag(new_imag) @ eigvecs^H
    # But we want F_c skew-Hermitian with eigenvalues 1j * new_imag,
    # which means the Hermitian matrix 1j*F_c has eigenvalues -new_imag.
    new_herm_eigvals = -new_imag
    F_c = -1j * (eigvecs * new_herm_eigvals[None, :]) @ eigvecs.conj().T

    n_lo = 1
    n_mid = F.shape[0] - 2
    n_hi = 1

    return F_c, (n_lo, n_mid, n_hi)


# ---------------------------------
# Real KKT vectorization (module-level)
# ---------------------------------
# Helpers for real embeddings of the Hermitian saddle matrix
#   M = [[A, V^H], [V, 0]]  on  z = [p; λ]  (complex, n = N²+m).
# ``CoastlinePoisson._kkt_*`` / MINRES use x = [Re z; Im z] ∈ R^{2n}.
# ``linear_operator`` uses block layout [Re p; Im p; Re λ; Im λ] (same map,
# permuted DOFs); it applies ``qf.laplacian.laplace`` on the (1,1) block while
# the class uses sparse ``A @ p`` for speed.


def real_vector_to_complex_matrix(x, N=None):
    """Stacked [Re(vec P); Im(vec P)] → complex N×N matrix P."""
    if N is None:
        N = int(round(np.sqrt(x.shape[0] / 2)))
    n2 = N * N
    return x[:n2].reshape(N, N) + 1j * x[n2 : 2 * n2].reshape(N, N)


def complex_matrix_to_real_vector(X):
    """Complex N×N matrix → stacked [Re(vec X); Im(vec X)]."""
    return np.concatenate([X.real.ravel(), X.imag.ravel()])


def real_vector_to_complex_vector(x, n_sq=None):
    """Stacked [Re(u); Im(u)] → complex vector u of length n_sq."""
    if n_sq is None:
        n_sq = x.shape[0] // 2
    return x[:n_sq] + 1j * x[n_sq : 2 * n_sq]


def complex_vector_to_real_vector(z):
    """Complex vector z → stacked [Re(z); Im(z)]."""
    return np.concatenate([z.real, z.imag])


def linear_operator(x, N, V):
    """
    Real matvec for the KKT operator with (1,1) block ``qf.laplacian.laplace``
    (dense matrix form, same operator as sparse ``A`` in ``CoastlinePoisson``).

    **Layout.** ``x`` stacks blocks ``[Re p; Im p; Re λ; Im λ]`` (length ``2(N²+m)``).
    This differs from ``CoastlinePoisson._kkt_real_matvec``, which uses the
    SciPy Hermitian embedding ``[Re z; Im z]`` for ``z = [p; λ]`` as one
    complex vector (permutation of entries only).

    Parameters
    ----------
    x : ndarray, shape (2*N**2 + 2*m,)
    N : int
    V : sparse or ndarray, shape (m, N²)
        Constraint rows (augmented ``V`` including trace row if desired).
    """
    n2 = N * N
    m = V.shape[0]
    x1 = x[: 2 * n2]
    x2 = x[2 * n2 :]
    X = real_vector_to_complex_matrix(x1, N)
    Lp = qf.laplacian.laplace(X)
    laplace_part = np.concatenate([Lp.real.ravel(), Lp.imag.ravel()])
    p_vec = real_vector_to_complex_vector(x1, n2)
    lam_vec = real_vector_to_complex_vector(x2, m)
    VH = V.conj().T
    constraint_part1 = complex_vector_to_real_vector(VH @ lam_vec)
    constraint_part2 = complex_vector_to_real_vector(V @ p_vec)
    return np.concatenate([laplace_part + constraint_part1, constraint_part2])


# ---------------------------------
# Constrained Poisson solver class
# ---------------------------------

class CoastlinePoisson:
    """
    Constrained Poisson solver:  -Delta_N P = W,  [F_c, P] = 0.

    Parameters
    ----------
    F_c : ndarray, shape (N, N)
        Truncated coastline matrix (from `truncate_to_level_set`).
    N : int
        Matrix bandwidth.
    method : str, "lu", "qr", "qr-simple", "im", "im-matrix", "qd", or "svd"
        How to extract the constraint rows.
    each_eigenvector : bool
        If True, build the constraint matrix from the nonzero-eigenvalue
        rank-1 projectors of F_c rather than from F_c itself.
    solver : str, "direct", "schur", "schur-poisson", "schur_cg", or "minres"
        "direct"   -- sparse LU of the full KKT saddle-point system.
        "schur"    -- explicit Schur complement (dense m x m LU factor).
        "schur-poisson" -- Schur complement with solve_poisson in the
                           repeated inverse-Laplacian applications.
        "schur_cg" -- CG on the Schur complement (preconditioned).
        "minres"   -- MINRES on the full KKT system (real 2n form).
    cg_tol : float
        Iterative solver tolerance (solver="schur_cg"/"minres").
    cg_maxiter : int or None
        Maximum iterations for the iterative solver.  None lets
        scipy choose a default.
    verbose : bool
        Print diagnostics during setup.
    """

    def __init__(self, F_c, N=None, method="lu", solver="direct",
                 each_eigenvector=False,
                 cg_tol=1e-12, cg_maxiter=None, verbose=True):
        if N is None:
            N = F_c.shape[0]
        self.N = N
        self.method = method
        self.solver_type = solver
        self.each_eigenvector = each_eigenvector
        self.solve_count = 0
        self.setup_diagnostics = {}
        self.last_solve_diagnostics = {}

        # Predicted rank
        rank, multiplicities = rank_from_eigenvalues(F_c)
        if verbose:
            print(f"N = {N},  rank(C) = {rank},  "
                  f"eigenvalue group sizes = {multiplicities}")

        if method == "im" and not each_eigenvector:
            self.setup_diagnostics["constraint_matrix"] = {
                "type": "commutator_linear_operator",
                "shape": (N**2, N**2),
            }
            V = self._build_constraint_im(F_c, rank)
        else:
            if each_eigenvector:
                C = self._build_constraint_matrix_each_eigenvector(F_c, rank)
                self.setup_diagnostics["constraint_matrix"] = {
                    "type": "each_eigenvector",
                    "shape": tuple(C.shape),
                }
            else:
                C = commutator_matrix(F_c, N)
                self.setup_diagnostics["constraint_matrix"] = {
                    "type": "commutator",
                    "shape": tuple(C.shape),
                }

            # Build constraint rows V (full row rank, same null space as C)
            if method == "lu":
                V = self._build_constraint_lu(C, rank)
            elif method == "qr":
                V = self._build_constraint_qr(C, rank)
            elif method == "qr-simple":
                V = self._build_constraint_qr_simple(C, rank)
            elif method == "im":
                V = self._build_constraint_im_from_matrix(C, rank)
            elif method == "im-matrix":
                V = self._build_constraint_im_matrix(C, rank)
            elif method == "qd":
                V = self._build_constraint_qd(C, rank)
            elif method == "svd":
                V = self._build_constraint_svd(C, rank)
            else:
                raise ValueError(
                    f"Unknown method '{method}'. "
                    "Use 'lu', 'qr', 'qr-simple', 'im', 'im-matrix', 'qd', or 'svd'."
                )

        V_nnz = int(V.nnz) if sp.issparse(V) else int(np.count_nonzero(V))
        V_total = int(V.shape[0] * V.shape[1])
        V_density = float(V_nnz / max(V_total, 1))
        self.setup_diagnostics["constraint_basis"] = {
            "method": method,
            "shape": tuple(V.shape),
            "nnz": V_nnz,
            "density": V_density,
        }
        if verbose:
            print(
                f"Constraint basis V ({method}) shape={V.shape}, "
                f"nnz={V_nnz}, density={V_density:.6e}"
            )

        if solver == "direct":
            self._init_direct(V, N, verbose)
        elif solver == "schur":
            self._init_schur(V, N, cg_tol, verbose)
        elif solver == "schur-poisson":
            self._init_schur(V, N, cg_tol, verbose)
        elif solver == "schur_cg":
            self._init_schur_cg(V, N, cg_tol, cg_maxiter, verbose)
        elif solver == "minres":
            self._init_minres(V, N, cg_tol, cg_maxiter, verbose)
        else:
            raise ValueError(
                f"Unknown solver '{solver}'. "
                "Use 'direct', 'schur', 'schur-poisson', 'schur_cg', or 'minres'."
            )

    def _init_direct(self, V, N, verbose):
        """Set up sparse LU factorization of the full KKT system."""
        A = qf.laplacian.sparse.laplacian(N, bc=False)

        # Add trace constraint tr(P) = 0 to fix the gauge freedom.
        # Without this, ker(A) ∩ ker(V) ≠ {0} (scalar matrices cI commute
        # with everything AND are in the null space of the Laplacian).
        trace_row = sp.csr_matrix(
            (np.ones(N), (np.zeros(N, dtype=int), np.arange(N) * (N + 1))),
            shape=(1, N**2)
        )
        V = sp.vstack([V, trace_row], format="csr")

        self.V = V
        self.n_constraints = V.shape[0]

        self.M = sp.bmat([
            [A, V.conj().T],
            [V, None]
        ], format="csc")

        if verbose:
            print(f"Factoring KKT system of size {self.M.shape[0]} ...")
        self._lu = spla.splu(self.M)
        if verbose:
            print("Done.")

    def _init_schur(self, V, N, cg_tol, verbose):
        """Set up Schur complement solver.

        Uses the Laplacian with BC (A_bc, nonsingular) as the inner
        operator when assembling the Schur complement.  The Schur
        complement is  S = V A_bc^{-1} V^H, formed explicitly and
        LU-factored.  No trace row is added to V; instead tr(P) = 0 is
        enforced after the solve (adding cI does not affect [F_c, P]
        since [F_c, I] = 0).
        """
        self.V = V if sp.issparse(V) else sp.csr_matrix(V)
        self.n_constraints = V.shape[0]
        m = self.n_constraints

        # Factor the Laplacian WITH BC (nonsingular)
        A_bc = qf.laplacian.sparse.laplacian(N, bc=True)
        self._A_lu = spla.splu(A_bc)

        # Form -S explicitly: (-S)_{:,j} = -V @ A_bc^{-1}(V^H e_j)
        VH = self.V.conj().T.tocsc()
        if verbose:
            print(f"Forming {m} x {m} Schur complement "
                  f"({m} solves) ...")
        neg_S = np.zeros((m, m), dtype=complex)
        for j in range(m):
            col_j = VH[:, j].toarray().ravel()
            z = self._A_lu.solve(col_j)
            neg_S[:, j] = -(self.V @ z)

        self._record_schur_setup_diagnostics(neg_S)
        self._schur_factor = scipy.linalg.lu_factor(neg_S)
        if verbose:
            print("Done.")

    def _record_schur_setup_diagnostics(self, neg_S):
        """Store condition diagnostics for the explicit Schur complement."""
        svals = scipy.linalg.svdvals(neg_S)
        schur_norm = np.linalg.norm(neg_S)
        hermitian_defect = np.linalg.norm(neg_S - neg_S.conj().T)
        row_norms = np.linalg.norm(neg_S, axis=1)
        sigma_max = float(np.max(svals)) if svals.size else 0.0
        sigma_min = float(np.min(svals)) if svals.size else 0.0
        cond_2 = np.inf if sigma_min == 0.0 else sigma_max / sigma_min
        self.setup_diagnostics["schur"] = {
            "size": int(neg_S.shape[0]),
            "sigma_max": sigma_max,
            "sigma_min": sigma_min,
            "cond_2": float(cond_2),
            "hermitian_defect_rel": float(
                hermitian_defect / max(schur_norm, 1e-30)
            ),
            "row_norm_min": float(np.min(row_norms)) if row_norms.size else 0.0,
            "row_norm_max": float(np.max(row_norms)) if row_norms.size else 0.0,
        }

    def _update_last_solve_diagnostics(self, stats):
        """Store diagnostics from the most recent linear solve."""
        self.solve_count += 1
        stats = dict(stats)
        stats["solve_count"] = self.solve_count
        self.last_solve_diagnostics = stats

    def _init_schur_cg(self, V, N, cg_tol, cg_maxiter, verbose):
        """Set up iterative Schur complement solver.

        Builds a symmetric negative definite regularization of A
        by adding  -(1/N) vec(I) vec(I)^T  (lifting the null-space
        eigenvalue from 0 to -1).  The resulting A_reg is symmetric,
        so S = V A_reg^{-1} V^H is Hermitian and CG applies.
        Each CG iteration costs one sparse triangular solve.
        """
        self.V = V if sp.issparse(V) else sp.csr_matrix(V)
        self.n_constraints = V.shape[0]
        self.cg_tol = cg_tol
        self.cg_maxiter = cg_maxiter

        A = qf.laplacian.sparse.laplacian(N, bc=False)

        # Symmetric rank-1 regularization: A_reg = A - (1/N) t t^T
        # where t = vec(I).  This makes A_reg negative definite.
        diag_idx = np.arange(N) * (N + 1)
        rows = np.repeat(diag_idx, N)
        cols = np.tile(diag_idx, N)
        vals = -np.ones(N**2, dtype=complex) / N
        correction = sp.csr_matrix((vals, (rows, cols)), shape=(N**2, N**2))
        A_reg = (A + correction).tocsc()

        self._A_lu = spla.splu(A_reg)
        self._last_lambda = None

        # Diagonal preconditioner: M ≈ -V diag(A_reg)^{-1} V^H.
        # Since diag(A_reg) is negative, -V diag(A_reg)^{-1} V^H is HPD,
        # matching the sign of -S.  Formed cheaply from sparse V.
        m = self.n_constraints
        d_inv = 1.0 / A_reg.diagonal()
        V_scaled = self.V.multiply(d_inv[np.newaxis, :])  # rows scaled by d^{-1}
        M_precond = -(V_scaled @ self.V.conj().T).toarray()
        self._precond_lu = scipy.linalg.lu_factor(M_precond)

        if verbose:
            print(f"Schur CG solver ready (m = {m}, "
                  f"symmetric A_reg factored, diagonal preconditioner built).")

    def _init_minres(self, V, N, tol, maxiter, verbose):
        """Set up MINRES solver on the full KKT saddle-point system.

        Uses the **same** KKT matrix as ``solver="direct"``: unregularized
        Laplacian ``A`` (``bc=False``) plus trace row in ``V``.  Do **not**
        use ``A_reg`` here — that changes the first block and produces a
        different ``(p, λ)``, which breaks the commutator constraint and
        skew-Hermiticity of ``P``.

        The KKT matrix  M = [A  V^H;  V  0]  is Hermitian.  MINRES
        handles indefiniteness.  scipy requires a real system, so the
        complex vector of length n is stacked as [Re z; Im z] (length 2n).

        Each iteration is sparse matvecs only (no factorization).
        """
        # Same (1,1) block as _init_direct — not A_reg.
        A = qf.laplacian.sparse.laplacian(N, bc=False).tocsc()

        trace_row = sp.csr_matrix(
            (np.ones(N), (np.zeros(N, dtype=int), np.arange(N) * (N + 1))),
            shape=(1, N**2)
        )
        V_aug = sp.vstack([V, trace_row], format="csr")

        self.V = V_aug
        self.n_constraints = V_aug.shape[0]
        self._A_kkt = A
        self.cg_tol = tol
        # Unpreconditioned MINRES on this KKT needs many Lanczos steps;
        # restarting (multiple short minres calls) is much worse than one
        # long run.  Default to a large iteration cap.
        n = N**2 + self.n_constraints
        if maxiter is not None:
            self.cg_maxiter = maxiter
        else:
            # ~40n matvecs is often needed for rtol ~1e-10 at moderate N.
            self.cg_maxiter = min(max(100_000, 40 * n), 500_000)
        self._last_x_real = None

        self._kkt_n = n
        self._build_minres_preconditioner()

        if verbose:
            m = self.n_constraints
            print(f"MINRES solver ready (N²={N**2}, m={m}, "
                  f"real system 2×{n}={2*n}, maxiter={self.cg_maxiter}).")

    def _kkt_apply_complex(self, z):
        """Apply Hermitian KKT ``M = [[A, V^H], [V, 0]]`` to ``z = [p; λ]``."""
        N = self.N
        n2 = N * N
        n = self._kkt_n
        if z.shape[0] != n:
            raise ValueError(f"expected z of length {n}, got {z.shape[0]}")
        p_c = z[:n2]
        lam_c = z[n2:]
        A = self._A_kkt
        V = self.V
        VH = V.conj().T
        y_p = A @ p_c + VH @ lam_c
        y_lam = V @ p_c
        return np.concatenate([y_p, y_lam])

    def _kkt_real_matvec(self, x):
        """Real embedding of ``M``: ``x = [Re z; Im z]`` with ``z ∈ C^n``."""
        n = self._kkt_n
        z = x[:n] + 1j * x[n:]
        y = self._kkt_apply_complex(z)
        return np.concatenate([y.real, y.imag])

    def _kkt_rhs_real_from_W(self, W):
        """Right-hand side ``[w; 0]`` in the same real stacking as ``_kkt_real_matvec``."""
        N = self.N
        m = self.n_constraints
        w = W.ravel()
        rhs_complex = np.concatenate([w, np.zeros(m, dtype=complex)])
        return np.concatenate([rhs_complex.real, rhs_complex.imag])

    def _build_minres_preconditioner(self):
        """Build block-diagonal preconditioner for MINRES.

        Approximate inverse of the KKT matrix:
            [ A   V^H ]^{-1}  ≈  [ A_bc^{-1}      0 ]
            [ V    0  ]        [    0       alpha^{-1} I ]

        where ``A_bc^{-1}`` is applied through ``qf.laplacian.solve_poisson``.
        """
        # Simple scale for multiplier block so lambda components are not
        # wildly mismatched to stream-function components.
        row_norm_sq = np.array(self.V.multiply(self.V.conj()).sum(axis=1)).real.ravel()
        alpha = float(np.sqrt(np.median(row_norm_sq[row_norm_sq > 0]))) if np.any(row_norm_sq > 0) else 1.0
        self._minres_lambda_scale = max(alpha, 1e-12)

        n = self._kkt_n
        self._minres_M_op = spla.LinearOperator(
            shape=(2 * n, 2 * n),
            matvec=self._minres_precond_real_matvec,
            dtype=float,
        )

    def _minres_precond_real_matvec(self, r):
        """Apply real-form preconditioner used by MINRES."""
        N = self.N
        n = self._kkt_n
        n2 = N * N
        z = r[:n] + 1j * r[n:]
        rp = z[:n2]
        rlam = z[n2:]

        # Fast approximate inverse Laplacian from quflow's optimized backend.
        # MINRES requires SPD preconditioner. qf.solve_poisson behaves like
        # an inverse negative Laplacian in this convention, so flip sign.
        yp = -qf.laplacian.solve_poisson(rp.reshape(N, N), bc=True).ravel()
        ylam = rlam / self._minres_lambda_scale
        y = np.concatenate([yp, ylam])
        return np.concatenate([y.real, y.imag])

    def kkt_real_linear_operator(self):
        """``LinearOperator`` for MINRES debugging / external Krylov drivers."""
        n = self._kkt_n
        return spla.LinearOperator(
            shape=(2 * n, 2 * n),
            matvec=self._kkt_real_matvec,
            dtype=float,
        )

    @staticmethod
    def _build_constraint_lu(C, rank):
        """Select `rank` independent rows from C via QR-pivoted row selection.

        Strategy: pick candidate rows with the largest norms (oversampled),
        then use a dense column-pivoted QR on their transpose to identify
        exactly `rank` linearly independent rows.  The result stays sparse.
        """
        C_csr = C.tocsr()
        row_norms = np.sqrt(np.array(C_csr.multiply(C_csr).sum(axis=1)).ravel())

        n_candidates = min(3 * rank, C.shape[0])
        candidate_idx = np.argsort(row_norms)[::-1][:n_candidates]

        C_sub = C_csr[candidate_idx].toarray()
        _, _, perm = scipy.linalg.qr(C_sub.T, pivoting=True, mode='economic')

        selected = candidate_idx[perm[:rank]]
        return C_csr[selected, :]

    @staticmethod
    def _build_constraint_qr(C, rank):
        """Select a sparse, better-scaled row basis for the constraint.

        This method keeps actual rows of ``C`` (hence stays sparse), but it
        improves over ``method="lu"`` in two ways:

        1. It oversamples more aggressively from the nonzero rows of ``C`` and
           mixes large-norm rows with rows spread across the spectrum of norms.
        2. It performs the pivoted QR on row-equilibrated candidates so the
           selection is driven more by linear independence than by raw scale.

        The returned rows are then normalized to unit row norm. Row scaling
        preserves the null space, so the constrained subspace is unchanged.
        """
        C_csr = C.tocsr()
        row_norms = np.sqrt(np.array(C_csr.multiply(C_csr.conj()).sum(axis=1)).real.ravel())
        nonzero_idx = np.flatnonzero(row_norms > 0)
        if nonzero_idx.size < rank:
            raise ValueError(
                f"Constraint matrix has only {nonzero_idx.size} nonzero rows, "
                f"cannot extract rank {rank}."
            )

        sorted_idx = nonzero_idx[np.argsort(row_norms[nonzero_idx])[::-1]]
        n_candidates = min(max(8 * rank, rank + 64), sorted_idx.size)
        n_top = min(2 * rank, n_candidates)
        top_idx = sorted_idx[:n_top]

        n_spread = n_candidates - n_top
        if n_spread > 0:
            spread_pos = np.linspace(0, sorted_idx.size - 1, n_spread, dtype=int)
            spread_idx = sorted_idx[spread_pos]
            candidate_idx = np.unique(np.concatenate([top_idx, spread_idx]))
        else:
            candidate_idx = np.unique(top_idx)

        # If deduplication reduced the pool too much, top it back up.
        if candidate_idx.size < rank:
            extra_needed = min(rank - candidate_idx.size, sorted_idx.size - candidate_idx.size)
            mask = np.ones(sorted_idx.size, dtype=bool)
            mask[np.isin(sorted_idx, candidate_idx)] = False
            extra = sorted_idx[mask][:extra_needed]
            candidate_idx = np.concatenate([candidate_idx, extra])

        C_sub = C_csr[candidate_idx, :].copy()
        sub_norms = row_norms[candidate_idx]
        scales = 1.0 / np.maximum(sub_norms, 1e-30)
        C_sub = C_sub.multiply(scales[:, np.newaxis])

        _, _, perm = scipy.linalg.qr(C_sub.toarray().T, pivoting=True, mode="economic")
        selected = candidate_idx[perm[:rank]]

        V = C_csr[selected, :].copy()
        V_norms = np.sqrt(np.array(V.multiply(V.conj()).sum(axis=1)).real.ravel())
        V = V.multiply((1.0 / np.maximum(V_norms, 1e-30))[:, np.newaxis])
        return V

    @staticmethod
    def _build_constraint_qr_simple(C, rank):
        """Select sparse rows via direct pivoted QR on all nonzero rows.

        This is the minimal QR-based baseline: no oversampling and no
        candidate preselection, only row equilibration followed by pivoted QR
        on ``C_nz^H`` and extraction of the corresponding original sparse rows.
        """
        C_csr = C.tocsr()
        row_norms = np.sqrt(np.array(C_csr.multiply(C_csr.conj()).sum(axis=1)).real.ravel())
        nonzero_idx = np.flatnonzero(row_norms > 0)
        if nonzero_idx.size < rank:
            raise ValueError(
                f"Constraint matrix has only {nonzero_idx.size} nonzero rows, "
                f"cannot extract rank {rank}."
            )

        C_nz = C_csr[nonzero_idx, :]
        scales = 1.0 / np.maximum(row_norms[nonzero_idx], 1e-30)
        C_eq = C_nz.multiply(scales[:, np.newaxis])
        _, _, perm = scipy.linalg.qr(C_eq.toarray().T, pivoting=True, mode="economic")
        selected = nonzero_idx[np.asarray(perm[:rank], dtype=int)]

        V = C_csr[selected, :].copy()
        V_norms = np.sqrt(np.array(V.multiply(V.conj()).sum(axis=1)).real.ravel())
        V = V.multiply((1.0 / np.maximum(V_norms, 1e-30))[:, np.newaxis])
        return V

    @staticmethod
    def _build_constraint_im_from_matrix(C, rank):
        """Select a sparse row basis via explicit-matrix interpolative decomposition.

        This applies ID directly to the adjoint of the explicitly constructed
        constraint matrix and uses deterministic pivoting (``rand=False``).
        """
        C_csr = C.tocsr()
        row_norms = np.sqrt(
            np.array(C_csr.multiply(C_csr.conj()).sum(axis=1)).real.ravel()
        )
        nonzero_idx = np.flatnonzero(row_norms > 0)
        if nonzero_idx.size < rank:
            raise ValueError(
                f"Constraint matrix has only {nonzero_idx.size} nonzero rows, "
                f"cannot extract rank {rank}."
            )

        C_nz = C_csr[nonzero_idx, :].copy()
        C_nz_H = C_nz.conj().transpose().tocsr()
        A = spla.LinearOperator(
            shape=C_nz_H.shape,
            matvec=lambda x: C_nz_H.dot(x),
            rmatvec=lambda x: C_nz_H.conj().transpose().dot(x),
            dtype=C_nz_H.dtype,
        )

        idx, _ = scipy.linalg.interpolative.interp_decomp(A, rank, rand=False)
        selected = nonzero_idx[np.asarray(idx[:rank], dtype=int)]

        V = C_csr[selected, :].copy()
        V_norms = np.sqrt(np.array(V.multiply(V.conj()).sum(axis=1)).real.ravel())
        V = V.multiply((1.0 / np.maximum(V_norms, 1e-30))[:, np.newaxis])
        return V

    @staticmethod
    def _build_constraint_im_matrix(C, rank):
        """Explicit-matrix ID path for comparing against matrix-free ``method="im"``."""
        return CoastlinePoisson._build_constraint_im_from_matrix(C, rank)

    @staticmethod
    def _build_constraint_im(F_c, rank):
        """Select a sparse row basis via matrix-free interpolative decomposition.

        The ID is applied to the adjoint commutator operator ``C^H`` without
        forming the Kronecker commutator matrix ``C`` explicitly. Skeleton
        column indices of ``C^H`` are then converted into explicit sparse rows
        of ``C`` using the row/column structure of the commutator.
        """
        selected = interpolative_constraint_indices(F_c, rank)
        V = selected_commutator_rows(F_c, selected)
        V_norms = np.sqrt(np.array(V.multiply(V.conj()).sum(axis=1)).real.ravel())
        V = V.multiply((1.0 / np.maximum(V_norms, 1e-30))[:, np.newaxis])
        return V

    @staticmethod
    def _build_constraint_qd(C, rank):
        """Select sparse rows using a Q-DEIM pivoting strategy.

        A rank-``rank`` orthonormal basis ``U`` for the row space of ``C`` is
        first computed from ``C^H``. Pivoted QR on ``U^T`` then selects row
        indices, which are mapped back to actual sparse rows of ``C``.
        """
        C_csr = C.tocsr()
        row_norms = np.sqrt(
            np.array(C_csr.multiply(C_csr.conj()).sum(axis=1)).real.ravel()
        )
        nonzero_idx = np.flatnonzero(row_norms > 0)
        if nonzero_idx.size < rank:
            raise ValueError(
                f"Constraint matrix has only {nonzero_idx.size} nonzero rows, "
                f"cannot extract rank {rank}."
            )

        C_nz = C_csr[nonzero_idx, :].copy()
        if C_nz.shape[0] == rank:
            V = C_nz
        else:
            U, svals, _ = spla.svds(C_nz.conj().transpose(), k=rank)
            sort_idx = np.argsort(svals)[::-1]
            U = U[:, sort_idx]
            _, _, perm = scipy.linalg.qr(U.conj().T, pivoting=True, mode="economic")
            selected = nonzero_idx[np.asarray(perm[:rank], dtype=int)]
            V = C_csr[selected, :].copy()

        V_norms = np.sqrt(np.array(V.multiply(V.conj()).sum(axis=1)).real.ravel())
        V = V.multiply((1.0 / np.maximum(V_norms, 1e-30))[:, np.newaxis])
        return V

    @staticmethod
    def _build_constraint_matrix_each_eigenvector(F_c, rank, tol=1e-10):
        """Build constraint matrix from the nonzero eigenprojectors of F_c.

        For each eigenvector ``e_i`` with nonzero eigenvalue, impose
        ``[P, e_i e_i^H] = 0``. In the eigenbasis of ``F_c``, this means all
        matrix entries touching the selected index ``i`` vanish, except the
        diagonal entry ``P_ii``. The resulting constraint has rank
        ``K(2N-K-1)`` when exactly ``K`` eigenvalues are nonzero.

        The returned matrix is already full row rank and spans the intended
        constraint row space. It can therefore be fed into ``lu``/``qr``/``svd``
        as the constraint matrix ``C``.
        """
        H = 1j * F_c
        evals, evecs = np.linalg.eigh(H)
        selected = np.flatnonzero(np.abs(evals) > tol)
        K = selected.size
        N = F_c.shape[0]
        expected_rank = K * (2 * N - K - 1)
        if expected_rank != rank:
            raise ValueError(
                f"Expected rank {rank}, but nonzero-eigenvalue projector "
                f"construction gives rank {expected_rank} (K={K})."
            )

        complement = np.setdiff1d(np.arange(N), selected, assume_unique=True)
        rows = []

        # Couplings between selected eigendirections and the complement.
        for i in selected:
            ei = evecs[:, i]
            for j in complement:
                ej = evecs[:, j]
                rows.append(np.outer(np.conj(ei), ej).ravel())
                rows.append(np.outer(np.conj(ej), ei).ravel())

        # Off-diagonal couplings within the selected eigendirections.
        for pos, i in enumerate(selected):
            ei = evecs[:, i]
            for j in selected[pos + 1 :]:
                ej = evecs[:, j]
                rows.append(np.outer(np.conj(ei), ej).ravel())
                rows.append(np.outer(np.conj(ej), ei).ravel())

        if len(rows) != rank:
            raise RuntimeError(
                f"Constructed {len(rows)} per-eigenvalue constraints, expected {rank}."
            )

        C = np.vstack(rows)
        row_norms = np.linalg.norm(C, axis=1)
        C /= np.maximum(row_norms[:, np.newaxis], 1e-30)
        return sp.csr_matrix(C)

    @staticmethod
    def _build_constraint_svd(C, rank):
        """Extract constraint basis via truncated SVD (dense result)."""
        if C.shape[0] <= rank:
            # C is already a full row-rank constraint matrix; return an
            # orthonormal row basis for the same row space.
            Q, _ = scipy.linalg.qr(C.toarray().T, mode="economic")
            return sp.csr_matrix(Q.conj().T)
        _, S, Vt = spla.svds(C, k=rank)
        sort_idx = np.argsort(S)[::-1]
        Vt = Vt[sort_idx]
        return sp.csr_matrix(Vt)

    def solve(self, W):
        """
        Solve the constrained Poisson equation for a given vorticity W.

        Parameters
        ----------
        W : ndarray, shape (N, N) or (k, N, N)
            Vorticity matrix (skew-Hermitian).

        Returns
        -------
        P : ndarray, shape (N, N) or (k, N, N)
            Stream matrix satisfying -Delta P = W and [F_c, P] = 0.
        """
        N = self.N
        single = (W.ndim == 2)
        if single:
            W = W[np.newaxis]

        results = []
        for Wi in W:
            if self.solver_type == "direct":
                P = self._solve_direct(Wi)
            elif self.solver_type == "schur":
                P = self._solve_schur(Wi)
            elif self.solver_type == "schur-poisson":
                P = self._solve_schur_poisson(Wi)
            elif self.solver_type == "schur_cg":
                P = self._solve_schur_cg(Wi)
            elif self.solver_type == "minres":
                P = self._solve_minres(Wi)
            else:
                P = self._solve_schur_cg(Wi)
            results.append(P)

        if single:
            return results[0]
        return np.array(results)

    def _solve_direct(self, W):
        """Solve via sparse LU of the full KKT system."""
        N = self.N
        rhs = np.concatenate([W.ravel(), np.zeros(self.n_constraints)])
        x = self._lu.solve(rhs)
        P = x[:N**2].reshape(N, N)
        self._update_last_solve_diagnostics({
            "solver": "direct",
            "rhs_norm": float(np.linalg.norm(rhs)),
            "solution_norm": float(np.linalg.norm(P)),
            "trace_before_projection": complex(np.trace(P)),
        })
        return P

    def _solve_schur(self, W):
        """Solve via pre-factored Schur complement.

        From  A_bc p + V^H lam = w,  V p = 0:
          p = A_bc^{-1}(w - V^H lam)
          (-S) lam = -V A_bc^{-1} w     [-S pre-factored via LU]
        Then tr(P) is subtracted (cI commutes with F_c).
        """
        N = self.N
        V = self.V

        p0 = self._A_lu.solve(W.ravel())
        b = V @ p0

        lam = scipy.linalg.lu_solve(self._schur_factor, -b)

        correction = V.conj().T @ lam
        p = self._A_lu.solve(W.ravel() - correction)
        constraint_residual = V @ p
        P = p.reshape(N, N)
        trace_before_projection = np.trace(P)
        P -= np.trace(P) / N * np.eye(N)
        self._update_last_solve_diagnostics({
            "solver": "schur",
            "rhs_norm": float(np.linalg.norm(W)),
            "p0_norm": float(np.linalg.norm(p0)),
            "schur_rhs_norm": float(np.linalg.norm(b)),
            "lambda_norm": float(np.linalg.norm(lam)),
            "correction_norm": float(np.linalg.norm(correction)),
            "constraint_residual_norm": float(np.linalg.norm(constraint_residual)),
            "constraint_residual_rel": float(
                np.linalg.norm(constraint_residual) / max(np.linalg.norm(p), 1e-30)
            ),
            "trace_before_projection": complex(trace_before_projection),
            "solution_norm": float(np.linalg.norm(P)),
        })
        return P

    def _solve_schur_poisson(self, W):
        """Schur solve variant using qf.laplacian.solve_poisson per RHS."""
        N = self.N
        V = self.V

        p0 = qf.laplacian.solve_poisson(W, bc=True).ravel()
        b = V @ p0

        lam = scipy.linalg.lu_solve(self._schur_factor, -b)

        correction = V.conj().T @ lam
        p = qf.laplacian.solve_poisson(
            (W.ravel() - correction).reshape(N, N),
            bc=True,
        ).ravel()
        constraint_residual = V @ p
        P = p.reshape(N, N)
        trace_before_projection = np.trace(P)
        P -= np.trace(P) / N * np.eye(N)
        self._update_last_solve_diagnostics({
            "solver": "schur-poisson",
            "rhs_norm": float(np.linalg.norm(W)),
            "p0_norm": float(np.linalg.norm(p0)),
            "schur_rhs_norm": float(np.linalg.norm(b)),
            "lambda_norm": float(np.linalg.norm(lam)),
            "correction_norm": float(np.linalg.norm(correction)),
            "constraint_residual_norm": float(np.linalg.norm(constraint_residual)),
            "constraint_residual_rel": float(
                np.linalg.norm(constraint_residual) / max(np.linalg.norm(p), 1e-30)
            ),
            "trace_before_projection": complex(trace_before_projection),
            "solution_norm": float(np.linalg.norm(P)),
        })
        return P

    def _solve_schur_cg(self, W):
        """Solve via CG on the Hermitian Schur complement.

        A_reg is symmetric negative definite, so S = V A_reg^{-1} V^H
        is Hermitian negative definite.  CG is applied to
        (-S) λ = -V A_reg^{-1} w  (positive definite system).

        Warm-started with the previous λ: consecutive calls (within
        the isomp fixed-point loop) have very similar W, so λ barely
        changes and CG converges in a few iterations.
        """
        N = self.N
        V = self.V
        m = self.n_constraints

        p0 = self._A_lu.solve(W.ravel())
        b = V @ p0

        def neg_schur_matvec(lam):
            y = V.conj().T @ lam
            z = self._A_lu.solve(y)
            return -(V @ z)

        neg_S_op = spla.LinearOperator(
            shape=(m, m), matvec=neg_schur_matvec, dtype=complex
        )

        def precond_solve(r):
            return scipy.linalg.lu_solve(self._precond_lu, r)

        M_op = spla.LinearOperator(
            shape=(m, m), matvec=precond_solve, dtype=complex
        )

        lam, info = spla.cg(
            neg_S_op, -b,
            x0=self._last_lambda,
            rtol=self.cg_tol,
            maxiter=self.cg_maxiter,
            M=M_op,
        )
        self._last_lambda = lam

        if info != 0:
            import warnings
            warnings.warn(
                f"CG did not converge (info={info}). "
                "Consider increasing cg_maxiter or relaxing cg_tol."
            )

        correction = V.conj().T @ lam
        p = self._A_lu.solve(W.ravel() - correction)
        constraint_residual = V @ p
        P = p.reshape(N, N)
        trace_before_projection = np.trace(P)
        P -= np.trace(P) / N * np.eye(N)
        self._update_last_solve_diagnostics({
            "solver": "schur_cg",
            "rhs_norm": float(np.linalg.norm(W)),
            "schur_rhs_norm": float(np.linalg.norm(b)),
            "lambda_norm": float(np.linalg.norm(lam)),
            "correction_norm": float(np.linalg.norm(correction)),
            "constraint_residual_norm": float(np.linalg.norm(constraint_residual)),
            "constraint_residual_rel": float(
                np.linalg.norm(constraint_residual) / max(np.linalg.norm(p), 1e-30)
            ),
            "trace_before_projection": complex(trace_before_projection),
            "solution_norm": float(np.linalg.norm(P)),
            "cg_info": int(info),
        })
        return P

    def _solve_minres(self, W):
        """Solve via MINRES on the real-form KKT system.

        Same Hermitian KKT as ``_init_direct``:  [A V^H; V 0] [p; λ] = [w; 0]
        with unregularized ``A``.  Converted to a real 2n system.
        """
        N = self.N
        n = self._kkt_n
        n2 = N**2

        rhs_real = self._kkt_rhs_real_from_W(W)
        kkt_op = self.kkt_real_linear_operator()

        # SciPy MINRES uses an estimated stopping test; we still verify
        # the true relative residual afterward and warn if needed.
        x = self._last_x_real
        bnorm = np.linalg.norm(rhs_real)
        if bnorm == 0:
            x = np.zeros(2 * n)
            rel_res = 0.0
            info = 0
        else:
            x, info = spla.minres(
                kkt_op, rhs_real,
                x0=x,
                rtol=self.cg_tol,
                maxiter=self.cg_maxiter,
                M=self._minres_M_op,
            )
            rvec = rhs_real - self._kkt_real_matvec(x)
            rel_res = np.linalg.norm(rvec) / bnorm

        self._last_x_real = x

        if info < 0:
            import warnings
            warnings.warn(f"MINRES illegal input/breakdown (info={info}).")
        elif rel_res > self.cg_tol:
            import warnings
            warnings.warn(
                f"MINRES true relative residual {rel_res:.3e} > rtol "
                f"{self.cg_tol:.3e} after {self.cg_maxiter} iterations "
                f"(info={info}). Increase cg_maxiter, relax cg_tol, or use "
                "solver='direct' / 'schur'."
            )

        z = x[:n] + 1j * x[n:]
        P = z[:n2].reshape(N, N)
        trace_before_projection = np.trace(P)
        P -= np.trace(P) / N * np.eye(N)
        self._update_last_solve_diagnostics({
            "solver": "minres",
            "rhs_norm": float(np.linalg.norm(rhs_real)),
            "solution_norm": float(np.linalg.norm(P)),
            "trace_before_projection": complex(trace_before_projection),
            "rel_res": float(rel_res),
            "minres_info": int(info),
        })
        return P

    def solve_poisson(self, W):
        """Drop-in replacement for qf.solve_poisson (same signature)."""
        return self.solve(W)


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


def plot_with_countour(W,F, level_set):
    return qf.plot(W, contour_data=F, contours=[level_set -0.02,level_set,level_set +0.02],colorbar=True)

def plot_eigenvalues(F,greater_than=-np.inf, less_than=np.inf,):
    Fvals, E = np.linalg.eigh(-1j*F)
    a = [fval for fval in Fvals if fval >= greater_than and fval <= less_than]
    qf.plot(F, contours=a,colorbar=True)



def make_trace_free_on_U(X,F,coastline_center,coastline_epsilon):
    E = scipy.linalg.orth(1j*F, rcond=1e-4)
    Fvals, E = np.linalg.eigh(1j*F)
    Fvals *= -1
    threshold = coastline_center - coastline_epsilon
    k = np.max(np.where(Fvals >= threshold)) + 1
    Q = E[:, :k]
    QH = Q.conj().T
    U = E[:, k:]
    UH = U.conj().T
    fis = Fvals[:k]
    fisrange = Fvals[k:]

    XU = UH @ X @ U
    XU -= np.trace(XU) * np.eye(U.shape[1]) / U.shape[1]
    return U @ XU @ UH


def project(W,F,coastline_center,coastline_epsilon):
    return make_trace_free_on_U(W,F,coastline_center,coastline_epsilon)


def project_soft(W,F,coastline_center,coastline_epsilon, epsilon=1e-4):
    XpI = np.eye(W.shape[0]) - F * (1.0j / epsilon)
    Wtilde = solve_continuous_lyapunov(XpI, 2 * W)
    # Remove the trace only on the allowed U-subspace, so we do not
    # add a constant in the blocked Q-region where the projection is zero.
    return make_trace_free_on_U(Wtilde,F,coastline_center,coastline_epsilon)

def matrix_commutator(A,B):
    return A @ B - B @ A

def matrix_commutator_from_F_linear_operator(F_c):
    n_sq = F_c.shape[0] ** 2

    def _matvec(x):
        return matrix_commutator(F_c, x.reshape(F_c.shape)).ravel()

    def _rmatvec(x):
        X = x.reshape(F_c.shape)
        return matrix_commutator(F_c.conj().T, X).ravel()

    return spla.LinearOperator(
        shape=(n_sq, n_sq),
        matvec=_matvec,
        rmatvec=_rmatvec,
        dtype=F_c.dtype,
    )

def selected_commutator_rows(F_c, selected):
    """Return explicit sparse commutator rows indexed by ``selected``.

    Row ``(a, b)`` of the commutator map ``P -> F_c P - P F_c`` only touches
    entries ``(j, b)`` and ``(a, j)`` of ``P``. This lets us reconstruct the
    selected rows directly from row ``a`` and column ``b`` of ``F_c`` without
    ever building the full Kronecker matrix.
    """
    N = F_c.shape[0]
    n_sq = N * N
    selected = np.asarray(selected, dtype=int)
    rows = []
    cols = []
    vals = []

    for row_pos, idx in enumerate(selected):
        a = idx // N
        b = idx % N

        row_a = F_c[a, :]
        nz_row = np.flatnonzero(np.abs(row_a) > 0)
        rows.extend([row_pos] * nz_row.size)
        cols.extend((nz_row * N + b).tolist())
        vals.extend(row_a[nz_row].tolist())

        col_b = F_c[:, b]
        nz_col = np.flatnonzero(np.abs(col_b) > 0)
        rows.extend([row_pos] * nz_col.size)
        cols.extend((a * N + nz_col).tolist())
        vals.extend((-col_b[nz_col]).tolist())

    V = sp.coo_matrix((vals, (rows, cols)), shape=(selected.size, n_sq)).tocsr()
    V.sum_duplicates()
    V.eliminate_zeros()
    return V

def commutator_row_norms_from_F(F_c):
    """Return Euclidean norms of the commutator rows without forming C."""
    row_sq = np.sum(np.abs(F_c) ** 2, axis=1)
    col_sq = np.sum(np.abs(F_c) ** 2, axis=0)
    diag = np.diag(F_c)
    overlap = (
        -np.abs(diag)[:, np.newaxis] ** 2
        -np.abs(diag)[np.newaxis, :] ** 2
        + np.abs(diag[:, np.newaxis] - diag[np.newaxis, :]) ** 2
    )
    norms_sq = row_sq[:, np.newaxis] + col_sq[np.newaxis, :] + overlap
    norms_sq = np.maximum(norms_sq.real, 0.0)
    return np.sqrt(norms_sq).ravel()

def interpolative_constraint_from_matrix(C, rank):
    """Return skeleton row indices from an interpolative decomposition."""
    C_csr = C.tocsr()
    row_norms = np.sqrt(np.array(C_csr.multiply(C_csr.conj()).sum(axis=1)).real.ravel())
    nonzero_idx = np.flatnonzero(row_norms > 0)
    if nonzero_idx.size < rank:
        raise ValueError(
            f"Constraint matrix has only {nonzero_idx.size} nonzero rows, "
            f"cannot extract rank {rank}."
        )

    C_nz = C_csr[nonzero_idx, :]
    C_nz_H = C_nz.conj().transpose().tocsr()
    A = spla.LinearOperator(
        shape=C_nz_H.shape,
        matvec=lambda x: C_nz_H.dot(x),
        rmatvec=lambda x: C_nz_H.conj().transpose().dot(x),
        dtype=C_nz_H.dtype,
    )
    idx, _ = scipy.linalg.interpolative.interp_decomp(A, rank)
    return nonzero_idx[np.asarray(idx[:rank], dtype=int)]

def interpolative_constraint_indices(F_c, rank):
    """Return skeleton commutator-row indices via matrix-free ID."""
    A = matrix_commutator_from_F_linear_operator(F_c)
    row_norms = commutator_row_norms_from_F(F_c)
    nonzero_idx = np.flatnonzero(row_norms > 0)
    if nonzero_idx.size < rank:
        raise ValueError(
            f"Constraint matrix has only {nonzero_idx.size} nonzero rows, "
            f"cannot extract rank {rank}."
        )

    n_sq = F_c.shape[0] ** 2

    def _matvec(x):
        weighted = np.zeros(n_sq, dtype=A.dtype)
        weighted[nonzero_idx] = x
        return A.rmatvec(weighted)

    def _rmatvec(y):
        full = A.matvec(y)
        return full[nonzero_idx]

    AH = spla.LinearOperator(
        shape=(n_sq, nonzero_idx.size),
        matvec=_matvec,
        rmatvec=_rmatvec,
        dtype=A.dtype,
    )
    idx, _ = scipy.linalg.interpolative.interp_decomp(AH, rank)
    return nonzero_idx[np.asarray(idx[:rank], dtype=int)]

def interpolative_constraint(F_c, rank):
    """Return a sparse full-rank commutator constraint matrix from ``F_c``."""
    selected = interpolative_constraint_indices(F_c, rank)
    V = selected_commutator_rows(F_c, selected)
    V_norms = np.sqrt(np.array(V.multiply(V.conj()).sum(axis=1)).real.ravel())
    return V.multiply((1.0 / np.maximum(V_norms, 1e-30))[:, np.newaxis])

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
  - "svd"  : computes a truncated SVD of C to obtain a dense but
             well-conditioned constraint basis (slower, dense block).

Four solver strategies are available:
  - "direct"   : sparse LU factorization of the full KKT saddle-point
                  matrix.  Pairs naturally with method="lu" (sparse V
                  keeps fill-in low).
  - "schur"    : explicit Schur complement — forms and LU-factors the
                  dense m x m matrix S = V A_bc^{-1} V^H.  2 Poisson
                  solves per step; m Poisson solves at setup.
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
import quflow as qf


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


def truncate_to_level_set(F, center, epsilon):
    """
    Keep only the eigenvalues of F whose imaginary part falls in
    (center - epsilon, center + epsilon). All other eigenvalues
    are set to exactly zero.

    Parameters
    ----------
    F : ndarray, shape (N, N)
        Skew-Hermitian matrix (quantized function).
    center : float
        Target imaginary eigenvalue.
    epsilon : float
        Half-width of the band.

    Returns
    -------
    F_c : ndarray, shape (N, N)
        Truncated matrix.
    K : int
        Number of retained eigenvalues.
    """
    eigvals, eigvecs = np.linalg.eig(F)
    mask = (center - epsilon <= eigvals.imag) & (eigvals.imag < center + epsilon)
    K = int(np.sum(mask))
    eigvals_out = np.where(mask, eigvals, 0.0)
    F_c = (eigvecs * eigvals_out[None, :]) @ eigvecs.conj().T
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
    method : str, "lu", "qr", or "svd"
        How to extract the constraint rows.
    solver : str, "direct", "schur", "schur_cg", or "minres"
        "direct"   -- sparse LU of the full KKT saddle-point system.
        "schur"    -- explicit Schur complement (dense m x m LU factor).
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
                 cg_tol=1e-12, cg_maxiter=None, verbose=True):
        if N is None:
            N = F_c.shape[0]
        self.N = N
        self.method = method
        self.solver_type = solver
        self.solve_count = 0
        self.setup_diagnostics = {}
        self.last_solve_diagnostics = {}

        # Commutator matrix
        C = commutator_matrix(F_c, N)

        # Predicted rank
        rank, multiplicities = rank_from_eigenvalues(F_c)
        if verbose:
            print(f"N = {N},  rank(C) = {rank},  "
                  f"eigenvalue group sizes = {multiplicities}")

        # Build constraint rows V (full row rank, same null space as C)
        if method == "lu":
            V = self._build_constraint_lu(C, rank)
        elif method == "qr":
            V = self._build_constraint_qr(C, rank)
        elif method == "svd":
            V = self._build_constraint_svd(C, rank)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'lu', 'qr', or 'svd'.")

        if solver == "direct":
            self._init_direct(V, N, verbose)
        elif solver == "schur":
            self._init_schur(V, N, cg_tol, verbose)
        elif solver == "schur_cg":
            self._init_schur_cg(V, N, cg_tol, cg_maxiter, verbose)
        elif solver == "minres":
            self._init_minres(V, N, cg_tol, cg_maxiter, verbose)
        else:
            raise ValueError(
                f"Unknown solver '{solver}'. "
                "Use 'direct', 'schur', 'schur_cg', or 'minres'."
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
        operator.  The Schur complement is  S = V A_bc^{-1} V^H,
        formed explicitly and LU-factored.  No trace row is added
        to V; instead tr(P) = 0 is enforced after the solve (adding
        cI does not affect [F_c, P] since [F_c, I] = 0).
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
        yp = -qf.laplacian.solve_poisson(rp.reshape(N, N)).ravel()
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
    def _build_constraint_svd(C, rank):
        """Extract constraint basis via truncated SVD (dense result)."""
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

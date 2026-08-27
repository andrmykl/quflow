r"""Commutator constraints for the quantized Poisson equation.

See ``docs/constraints.md`` for construction modes and derivations.
"""

import numpy as np
import scipy.linalg
import scipy.linalg.interpolative
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import quflow as qf


__all__ = [
    "CommutatorPoissonSolver",
    "constraint_matrix",
    "spectral_matrix",
    "trace_free_block_projector",
]


_ROW_NORM_TOL = 1e-12


# Matrix-free commutator constraints


def _commutator_operator(F):
    """Return the matrix-free operator ``X -> bracket(F, X)``."""
    N = F.shape[0]
    N2 = N**2

    def matvec(X):
        return qf.geometry.bracket(F, np.asarray(X).reshape(N, N)).ravel()

    def rmatvec(X):
        return -matvec(X)

    return spla.LinearOperator(
        (N2, N2), matvec=matvec, rmatvec=rmatvec, dtype=F.dtype
    )


def _commutator_coordinate_row_norms(F):
    """Return all commutator coordinate-row norms without forming the matrix."""
    row_sq = np.sum(np.abs(F) ** 2, axis=1)
    col_sq = np.sum(np.abs(F) ** 2, axis=0)
    diag = np.diag(F)
    diag_sq = np.abs(diag) ** 2
    overlap = (
        -diag_sq[:, np.newaxis]
        - diag_sq[np.newaxis, :]
        + np.abs(diag[:, np.newaxis] - diag[np.newaxis, :]) ** 2
    )
    norms_sq = row_sq[:, np.newaxis] + col_sq[np.newaxis, :] + overlap
    return np.sqrt(np.maximum(norms_sq.real, 0.0)).ravel() / qf.geometry.hbar(
        F.shape[0]
    )


def _materialize_commutator_rows(F, selected_row_indices):
    """Return selected row-major commutator rows as a CSR matrix."""
    N = F.shape[0]
    N2 = N**2
    selected_row_indices = np.asarray(selected_row_indices, dtype=int)
    if selected_row_indices.size == 0:
        return sp.csr_matrix((0, N2), dtype=F.dtype)

    rows = []
    cols = []
    values = []

    # This is the sparse coefficient form of qf.geometry.bracket.  Calling
    # bracket on one dense basis matrix per selected row would be much costlier.
    for row_position, flat_index in enumerate(selected_row_indices):
        output_row = flat_index // N
        output_column = flat_index % N

        matrix_row = F[output_row, :]
        nonzero = np.flatnonzero(matrix_row)
        rows.extend([row_position] * nonzero.size)
        cols.extend((nonzero * N + output_column).tolist())
        values.extend(matrix_row[nonzero].tolist())

        matrix_column = F[:, output_column]
        nonzero = np.flatnonzero(matrix_column)
        rows.extend([row_position] * nonzero.size)
        cols.extend((output_row * N + nonzero).tolist())
        values.extend((-matrix_column[nonzero]).tolist())

    values = np.asarray(values, dtype=F.dtype) / qf.geometry.hbar(N)
    selected_rows = sp.coo_matrix(
        (values, (rows, cols)),
        shape=(selected_row_indices.size, N2),
    ).tocsr()
    selected_rows.sum_duplicates()
    selected_rows.eliminate_zeros()
    return selected_rows


# Spectral construction


def spectral_matrix(
    functions,
    level_sets,
    new_eigenvalues=None,
    *,
    superlevel=False,
    num_extra_eigenvectors=0,
    only_lowest=False,
    only_highest=False,
):
    """Select spectral blocks and assign new eigenvalues.

    By default, only the eigenvector closest to each level is selected.  With
    ``superlevel=True``, the selected block begins at that eigenvalue and
    includes all larger eigenvalues.  Positive ``num_extra_eigenvectors``
    extends the block downward, while negative values move its lower boundary
    upward.  ``only_lowest=True`` or ``only_highest=True`` retains only the
    corresponding endpoint eigenvector of the resulting block.  If
    ``new_eigenvalues`` is omitted, each block retains its closest eigenvalue.
    """
    functions = tuple(np.asarray(F) for F in functions)
    level_sets = np.asarray(level_sets, dtype=float).ravel()
    if not isinstance(num_extra_eigenvectors, int):
        raise ValueError("num_extra_eigenvectors must be an integer")
    if only_lowest and only_highest:
        raise ValueError("only_lowest and only_highest cannot both be True")
    if not functions:
        raise ValueError("functions and level_sets must not be empty.")
    if len(functions) != level_sets.size:
        raise ValueError("functions and level_sets must have equal lengths.")
    if not np.isfinite(level_sets).all():
        raise ValueError("level_sets must be finite real values.")

    if new_eigenvalues is None:
        new_eigenvalues = (None,) * len(functions)
    else:
        try:
            new_eigenvalues = tuple(new_eigenvalues)
        except TypeError:
            new_eigenvalues = (new_eigenvalues,) * len(functions)
        if len(new_eigenvalues) != len(functions):
            raise ValueError(
                "new_eigenvalues and functions must have equal lengths."
            )

    matrix = 0.0
    for function_index, (F, level, new_eigenvalue) in enumerate(
        zip(functions, level_sets, new_eigenvalues)
    ):
        eigenvalues, eigenvectors = np.linalg.eigh(-1j * F)
        closest_index = np.argmin(np.abs(eigenvalues - level))
        closest_eigenvalue = eigenvalues[closest_index]

        if superlevel:
            keep = np.isclose(eigenvalues, closest_eigenvalue)
            keep |= eigenvalues > closest_eigenvalue
        else:
            keep = np.zeros(eigenvalues.size, dtype=bool)
            keep[closest_index] = True

        if num_extra_eigenvectors > 0:
            lower_indices = np.flatnonzero(
                (eigenvalues < closest_eigenvalue) & ~keep
            )
            if num_extra_eigenvectors > lower_indices.size:
                raise ValueError(
                    f"num_extra_eigenvectors={num_extra_eigenvectors} exceeds "
                    f"the {lower_indices.size} eigenvalues below the closest "
                    f"eigenvalue of functions[{function_index}]."
                )
            keep[lower_indices[-num_extra_eigenvectors:]] = True
        elif num_extra_eigenvectors < 0:
            selected_indices = np.flatnonzero(keep)
            remove_count = -num_extra_eigenvectors
            if remove_count >= selected_indices.size:
                raise ValueError(
                    f"num_extra_eigenvectors={num_extra_eigenvectors} "
                    f"removes all {selected_indices.size} selected "
                    f"eigenvectors of functions[{function_index}]."
                )
            keep[selected_indices[:remove_count]] = False

        if only_lowest or only_highest:
            selected_indices = np.flatnonzero(keep)
            selected_index = (
                selected_indices[-1] if only_highest else selected_indices[0]
            )
            keep[:] = False
            keep[selected_index] = True

        U = eigenvectors[:, keep]
        value = (
            closest_eigenvalue if new_eigenvalue is None else new_eigenvalue
        )
        matrix = matrix + np.asarray(value)[..., None, None] * (U @ U.conj().T)

    return matrix


def constraint_matrix(
    functions,
    level_sets,
    keep_eigenvalues=False,
    *,
    superlevel=False,
    new_eigenvalues=None,
):
    """Construct a skew-Hermitian constraint matrix.

    Replacement eigenvalues default to distinct labels from 1 to 2.  In
    superlevel mode, every eigenvector from the closest eigenvalue upward gets
    the corresponding replacement value.  Set ``keep_eigenvalues=True`` to
    use the closest original eigenvalues instead.
    """
    functions = tuple(functions)
    if keep_eigenvalues and new_eigenvalues is not None:
        raise ValueError(
            "keep_eigenvalues and new_eigenvalues cannot be used together."
        )
    if not keep_eigenvalues and new_eigenvalues is None:
        new_eigenvalues = np.linspace(1.0, 2.0, len(functions))
    return 1j * spectral_matrix(
        functions,
        level_sets,
        new_eigenvalues,
        superlevel=superlevel,
    )


def trace_free_block_projector(
    functions,
    level_sets,
    outside_values,
    *,
    num_extra_eigenvectors=0,
    only_lowest=False,
    only_highest=False,
):
    """Create a trace-free projector with prescribed spectral outside blocks.

    Each outside block begins at the eigenvalue closest to its paired level
    and includes every larger eigenvalue.  Positive
    ``num_extra_eigenvectors`` extends the block downward; negative values
    move its boundary upward.  With ``only_lowest=True``, the prescribed value
    is placed only on the eigenvector selected by that boundary shift and the
    other outside eigenvectors are set to zero.  A negative shift does not
    return the skipped eigenvectors to the complementary block.
    ``only_highest=True`` similarly places the value only on the highest
    eigenvector.  The two options are mutually exclusive.  The returned
    callable removes trace through the complementary block.  Derived
    projectors are assumed pairwise orthogonal.
    """

    functions = tuple(np.asarray(F) for F in functions)
    level_sets = np.asarray(level_sets, dtype=float).ravel()
    outside_values = tuple(outside_values)
    if (
        len(functions) != level_sets.size
        or len(functions) != len(outside_values)
    ):
        raise ValueError(
            "functions, level_sets, and outside_values must have equal "
            "lengths."
        )
    if not np.isfinite(level_sets).all():
        raise ValueError("level_sets must be finite real values.")
    if only_lowest and only_highest:
        raise ValueError("only_lowest and only_highest cannot both be True")

    matrix_shape = functions[0].shape if functions else None
    if functions:
        projector_extra_eigenvectors = (
            max(num_extra_eigenvectors, 0)
            if only_lowest or only_highest
            else num_extra_eigenvectors
        )
        outside_projector = spectral_matrix(
            functions,
            level_sets,
            1.0,
            superlevel=True,
            num_extra_eigenvectors=projector_extra_eigenvectors,
        )
        outside_values = tuple(
            value if np.iscomplexobj(value) else 1j * np.asarray(value)
            for value in outside_values
        )
        outside_matrix = spectral_matrix(
            functions,
            level_sets,
            outside_values,
            superlevel=True,
            num_extra_eigenvectors=num_extra_eigenvectors,
            only_lowest=only_lowest,
            only_highest=only_highest,
        )
    else:
        outside_projector = 0.0
        outside_matrix = 0.0

    def project(W):
        W = np.asarray(W)
        if W.ndim < 2 or W.shape[-2] != W.shape[-1]:
            raise ValueError(
                f"W must end with square matrix axes, got {W.shape}."
            )
        if matrix_shape is not None and W.shape[-2:] != matrix_shape:
            raise ValueError(
                f"W must end with shape {matrix_shape}, got {W.shape}."
            )
        identity = np.eye(W.shape[-1], dtype=W.dtype)
        water_projector = identity - outside_projector
        projected = water_projector @ W @ water_projector + outside_matrix

        trace = np.trace(projected, axis1=-2, axis2=-1)
        water_rank = np.trace(water_projector)
        if np.isclose(water_rank, 0.0, rtol=1e-12, atol=1e-12):
            if np.allclose(trace, 0.0, rtol=1e-12, atol=1e-12):
                return projected
            raise ValueError(
                "Cannot preserve the outside values and make the result "
                "trace-free because the outside blocks span the full "
                "matrix space."
            )
        return projected - (trace / water_rank)[..., None, None] * water_projector

    return project


# Constrained Poisson solver


class CommutatorPoissonSolver:
    """Reusable Poisson solver enforcing ``bracket(F_c, P) == 0``.

    A positive active count ``K`` assumes ``K`` distinct nonzero eigenvalues
    and an ``(N-K)``-fold zero eigenvalue.  Pass zero to detect the rank
    numerically.  Construction briefly selects QuFlow's generic Poisson mode
    and is not thread-safe with concurrent Poisson calls.
    """

    def __init__(self, constraint_matrix, num_active_eigenvalues, rng=None):
        constraint_matrix = np.asarray(constraint_matrix)
        shape = constraint_matrix.shape
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(
                f"constraint_matrix must be square, got shape {shape}."
            )

        N = shape[0]
        count_error = (
            "num_active_eigenvalues must be an integer between 0 and "
            f"{N}, got {num_active_eigenvalues}."
        )
        try:
            K = int(num_active_eigenvalues)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(count_error) from error
        if (
            np.ndim(num_active_eigenvalues) != 0
            or K != num_active_eigenvalues
            or not 0 <= K <= N
        ):
            raise ValueError(count_error)

        self.matrix_size = N
        self.num_active_eigenvalues = K
        self.rng = np.random.default_rng(rng)
        self.constraint_rank = K * (2 * N - K - 1)
        self.constraint_row_basis = self._select_constraint_row_basis(
            constraint_matrix
        )
        num_rows = self.constraint_row_basis.shape[0]
        print(
            f"CommutatorPoissonSolver: retained {num_rows} constraint rows.",
            flush=True,
        )
        self._factor_schur_complement()

    def _select_constraint_row_basis(self, constraint_matrix):
        """Select normalized independent rows of the commutator."""
        N2 = self.matrix_size**2
        automatic_rank = self.num_active_eigenvalues == 0
        empty = sp.csr_matrix((0, N2), dtype=constraint_matrix.dtype)
        if self.constraint_rank == 0 and not automatic_rank:
            return empty

        commutator = _commutator_operator(constraint_matrix)
        norms = _commutator_coordinate_row_norms(constraint_matrix)
        max_norm = float(np.max(norms))
        if automatic_rank and max_norm == 0.0:
            return empty

        cutoff = _ROW_NORM_TOL * max_norm
        usable = norms > cutoff
        num_usable = int(np.count_nonzero(usable))
        if not automatic_rank and num_usable < self.constraint_rank:
            raise ValueError(
                f"Only {num_usable} commutator rows have norm above "
                f"{cutoff:.3e}, but the expected rank is {self.constraint_rank}."
            )

        inverse_norms = np.zeros_like(norms)
        inverse_norms[usable] = 1.0 / norms[usable]
        inverse_norm_operator = spla.aslinearoperator(
            sp.diags(inverse_norms, format="csr")
        )

        # Columns of C.H are conjugated rows of C.  Right scaling makes those
        # columns unit norm before interpolative decomposition.
        normalized_adjoint = commutator.H @ inverse_norm_operator
        if automatic_rank:
            detected_rank, pivots, _ = scipy.linalg.interpolative.interp_decomp(
                normalized_adjoint,
                _ROW_NORM_TOL,
                rng=self.rng,
            )
            self.constraint_rank = int(detected_rank)
        else:
            pivots, _ = scipy.linalg.interpolative.interp_decomp(
                normalized_adjoint,
                self.constraint_rank,
                rng=self.rng,
            )

        if num_usable < self.constraint_rank:
            raise RuntimeError(
                "Interpolative decomposition detected more independent rows "
                "than have norm above the row cutoff."
            )
        if self.constraint_rank == 0:
            return empty

        indices = np.asarray(pivots[: self.constraint_rank], dtype=int)
        if np.any(~usable[indices]):
            raise RuntimeError("Interpolative decomposition selected a zero row.")

        rows = _materialize_commutator_rows(constraint_matrix, indices)
        return rows.multiply(inverse_norms[indices, np.newaxis]).tocsr()

    def _factor_schur_complement(self):
        """Materialize and LU-factor ``-V A**-1 V*`` for reuse."""
        if self.constraint_rank == 0:
            self._schur_factorization = None
            return

        N = self.matrix_size
        rank = self.constraint_rank
        V = self.constraint_row_basis
        V_adjoint = V.conj().T.tocsr()
        dtype = np.result_type(V.dtype, np.complex128)
        schur = np.empty((rank, rank), dtype=dtype)

        previous_mode = qf.laplacian.cpu.select_skewherm(False)
        try:
            for column in range(rank):
                rhs = np.asarray(
                    V_adjoint[:, column].toarray(), dtype=dtype
                ).reshape(N, N)
                inverse_rhs = np.asarray(qf.solve_poisson(rhs), dtype=dtype)
                schur[:, column] = -(V @ inverse_rhs.ravel())
        finally:
            qf.laplacian.cpu.select_skewherm(previous_mode)

        self._schur_factorization = scipy.linalg.lu_factor(schur)

    def _solve_poisson_columns(self, rhs):
        """Apply ``qf.solve_poisson`` separately to vectorized columns."""
        solutions = np.empty(
            rhs.shape, dtype=np.result_type(rhs.dtype, np.complex128)
        )
        N = self.matrix_size
        for column in range(rhs.shape[1]):
            W = np.asarray(rhs[:, column], dtype=solutions.dtype).reshape(N, N)
            # Copy each result before qf.solve_poisson reuses its output cache.
            solutions[:, column] = np.asarray(qf.solve_poisson(W)).ravel()
        return solutions

    def solve(self, W):
        """Solve the constrained Poisson equation for a matrix or batch."""
        W = np.asarray(W)
        N = self.matrix_size
        if W.shape[-2:] != (N, N):
            raise ValueError(f"W must end with shape {(N, N)}, got {W.shape}.")

        rhs = W.reshape(-1, N**2).T
        unconstrained = self._solve_poisson_columns(rhs)

        if self.constraint_rank:
            V = self.constraint_row_basis
            schur_rhs = V @ unconstrained
            multipliers = scipy.linalg.lu_solve(
                self._schur_factorization, -schur_rhs
            )
            correction = V.conj().T @ multipliers
            solutions = self._solve_poisson_columns(rhs - correction)
        else:
            solutions = unconstrained

        P = solutions.T.reshape(W.shape)
        trace = np.trace(P, axis1=-2, axis2=-1)
        P -= (trace / N)[..., None, None] * np.eye(N)
        return P

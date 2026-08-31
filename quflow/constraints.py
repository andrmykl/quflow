r"""Commutator constraints for the quantized Poisson equation.

See ``docs/constraints.md`` for construction modes and derivations.
"""

from collections.abc import Mapping

import numpy as np
import scipy.linalg
import scipy.linalg.interpolative
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import quflow as qf


__all__ = [
    "CommutatorPoissonSolver",
    "ConstraintPlotter",
    "constraint_matrix",
    "plotter",
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


# Constraint-aware plotting


class ConstraintPlotter:
    """Plot states with fixed level-set contours and an optional subtraction.

    Instances are callable and return the image artist created by
    :func:`quflow.plot`.  The same instance can be supplied to
    :class:`quflow.Animation` through its ``plotter`` argument, which keeps
    the contours fixed while updating the state beneath them.

    ``subtract_function`` is interpreted as a fixed field.  If it is
    callable, it is evaluated once on a zero array shaped like the first
    constraint function.  This makes it possible to pass the affine callable
    returned by :func:`trace_free_block_projector`; only its constant part is
    removed, rather than applying the projector to every animation frame.

    Plotting uses at least ``min_N=256`` samples in latitude (and ``2*N-1``
    in longitude).  Above that floor, ``N`` is inherited from the plotted
    state unless supplied explicitly.  Set ``min_N=None`` to allow lower
    plotting resolutions.
    """

    def __init__(
        self,
        functions,
        level_sets,
        subtract_function=None,
        *,
        min_N=256,
        contour_kwargs=None,
        **plot_kwargs,
    ):
        self.functions = tuple(np.array(F, copy=True) for F in functions)
        self.level_sets = np.asarray(level_sets, dtype=float).ravel()
        if not self.functions:
            raise ValueError("functions and level_sets must not be empty.")
        if len(self.functions) != self.level_sets.size:
            raise ValueError("functions and level_sets must have equal lengths.")
        if not np.isfinite(self.level_sets).all():
            raise ValueError("level_sets must be finite real values.")

        reference = self.functions[0]
        if subtract_function is None:
            self.subtract_function = None
        else:
            if callable(subtract_function):
                subtract_function = subtract_function(np.zeros_like(reference))
            subtraction = np.asarray(subtract_function)
            if subtraction.shape != reference.shape:
                raise ValueError(
                    "subtract_function must have the same shape as "
                    f"functions[0], got {subtraction.shape} and "
                    f"{reference.shape}."
                )
            self.subtract_function = np.array(subtraction, copy=True)

        self.min_N = self._validate_resolution(min_N, "min_N", allow_none=True)
        self.contour_kwargs = self._normalize_contour_kwargs(contour_kwargs)
        self.plot_kwargs = dict(plot_kwargs)
        self._validate_plot_kwargs(self.plot_kwargs)
        if self.plot_kwargs.get("N") is not None:
            self._validate_resolution(self.plot_kwargs["N"], "N")
        self._contour_fun_cache = {}

    @staticmethod
    def _validate_resolution(value, name, *, allow_none=False):
        if value is None and allow_none:
            return None
        error_message = f"{name} must be a positive integer"
        if np.iscomplexobj(value):
            raise ValueError(error_message)
        try:
            resolution = int(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(error_message) from error
        if (
            np.ndim(value) != 0
            or isinstance(value, (bool, np.bool_))
            or resolution != value
            or resolution < 1
        ):
            raise ValueError(error_message)
        return resolution

    @staticmethod
    def _infer_resolution(state):
        """Infer spherical bandwidth from matrix, function, or coefficients."""
        state = np.asarray(state)
        if state.ndim == 2:
            if state.shape[0] < 1:
                raise ValueError("Cannot infer N from an empty state.")
            return state.shape[0]
        if state.ndim == 1 and state.size:
            return int(np.ceil(np.sqrt(state.size)))
        raise ValueError(
            "Cannot infer N: state must be a nonempty one- or two-dimensional "
            "array."
        )

    def _resolve_resolution(self, state, N=None):
        if N is None:
            N = self._infer_resolution(state)
        else:
            N = self._validate_resolution(N, "N")
        if self.min_N is not None:
            N = max(N, self.min_N)
        return N

    @staticmethod
    def _validate_plot_kwargs(plot_kwargs):
        reserved = {"contours", "contour_data", "contour_kwargs"}
        unsupported = reserved.intersection(plot_kwargs)
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(
                f"{names} cannot be used with ConstraintPlotter; configure "
                "the fixed level-set contours when constructing the plotter."
            )

    def _normalize_contour_kwargs(self, contour_kwargs):
        """Return one independent contour keyword dictionary per field."""
        if contour_kwargs is None:
            kwargs_per_contour = [{} for _ in self.functions]
        elif isinstance(contour_kwargs, Mapping):
            kwargs_per_contour = [
                dict(contour_kwargs) for _ in self.functions
            ]
        else:
            try:
                kwargs_per_contour = [dict(kwargs) for kwargs in contour_kwargs]
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "contour_kwargs must be a mapping or one mapping per "
                    "constraint function."
                ) from error
            if len(kwargs_per_contour) != len(self.functions):
                raise ValueError(
                    "A contour_kwargs sequence must have one entry per "
                    "constraint function."
                )

        for kwargs in kwargs_per_contour:
            if "levels" in kwargs:
                raise ValueError(
                    "contour levels are set by level_sets, not contour_kwargs."
                )
        return tuple(kwargs_per_contour)

    def prepare(self, state):
        """Return the state after removing the configured fixed field."""
        state = np.asarray(state)
        if self.subtract_function is None:
            return state
        try:
            return state - self.subtract_function
        except ValueError as error:
            raise ValueError(
                f"state shape {state.shape} is incompatible with subtraction "
                f"shape {self.subtract_function.shape}."
            ) from error

    def _contour_functions(self, N):
        """Convert and cache all contour fields at plotting resolution ``N``."""
        cache_key = None if N is None else int(N)
        if cache_key not in self._contour_fun_cache:
            contour_functions = []
            for field in self.functions:
                if N is not None:
                    field = qf.graphics.resample(field, N)
                contour_fun = qf.as_fun(field)
                if np.iscomplexobj(contour_fun):
                    contour_fun = contour_fun.real
                contour_functions.append(contour_fun)
            self._contour_fun_cache[cache_key] = tuple(contour_functions)
        return self._contour_fun_cache[cache_key]

    @staticmethod
    def _uses_cartopy(ax):
        """Return whether ``ax`` expects geographic data in degrees."""
        return qf.graphics._is_cartopy_axes(ax)

    def _draw_contours(self, ax, N, user_annotate=None):
        use_cartopy = self._uses_cartopy(ax)
        for contour_fun, level, user_kwargs in zip(
            self._contour_functions(N),
            self.level_sets,
            self.contour_kwargs,
        ):
            lon = np.linspace(
                -np.pi,
                np.pi,
                contour_fun.shape[1],
                endpoint=False,
            )
            lat = np.linspace(
                -np.pi / 2,
                np.pi / 2,
                contour_fun.shape[0],
            )
            kwargs = {
                "colors": "black",
                "linewidths": 1.0,
                "negative_linestyles": "solid",
            }
            kwargs.update(user_kwargs)
            kwargs["levels"] = [level]
            if use_cartopy:
                lon = np.rad2deg(lon)
                lat = np.rad2deg(lat)
                kwargs.setdefault("transform", qf.graphics.ccrs.PlateCarree())
            ax.contour(lon, lat, contour_fun, **kwargs)

        if user_annotate is not None:
            user_annotate(ax)

    def __call__(self, state, **plot_kwargs):
        """Plot ``state`` with the configured subtraction and contours."""
        kwargs = {**self.plot_kwargs, **plot_kwargs}
        self._validate_plot_kwargs(kwargs)
        N = self._resolve_resolution(state, kwargs.get("N"))
        kwargs["N"] = N
        user_annotate = kwargs.pop("annotate", None)
        im = qf.plot(self.prepare(state), **kwargs)
        im.axes.set_autoscale_on(False)
        self._draw_contours(im.axes, N, user_annotate=user_annotate)
        im._quflow_constraint_plotter_N = N
        return im

    def update(self, im, state, *, N=None):
        """Update ``im`` without redrawing contours or changing resolution."""
        image_N = getattr(im, "_quflow_constraint_plotter_N", None)
        if image_N is not None:
            if N is not None:
                requested_N = self._resolve_resolution(state, N)
                if requested_N != image_N:
                    raise ValueError(
                        f"Cannot update an image created with N={image_N} "
                        f"using N={requested_N}. Create a new image to change "
                        "plotting resolution."
                    )
            N = image_N
        elif N is None:
            N = self.plot_kwargs.get("N")
        N = self._resolve_resolution(state, N)

        data = self.prepare(state)
        if N is not None:
            data = qf.graphics.resample(data, N)
        fun = qf.as_fun(data)
        if np.iscomplexobj(fun):
            fun = fun.real

        if hasattr(im, "get_array") and np.size(im.get_array()) != fun.size:
            raise ValueError(
                "The updated state has a different plotting resolution from "
                "the existing image. Pass a consistent N to the plotter and "
                "Animation."
            )
        if hasattr(im, "set_data"):
            im.set_data(fun)
        elif hasattr(im, "set_array"):
            im.set_array(fun.ravel())
        else:
            raise AttributeError("Could not find method for setting data.")
        return im


def plotter(
    functions,
    level_sets,
    subtract_function=None,
    *,
    min_N=256,
    contour_kwargs=None,
    **plot_kwargs,
):
    """Return a reusable :class:`ConstraintPlotter`.

    The effective plotting bandwidth is ``max(256, N)`` by default.  If ``N``
    is omitted, it is inferred from each initial state.  Set ``min_N=None``
    to disable the default floor.

    Examples
    --------
    ``project`` may be the callable returned by
    :func:`trace_free_block_projector`::

        cplot = plotter(functions, level_sets, project, N=N)
        cplot(W0)

        with qf.Animation("simulation.mp4", plotter=cplot) as animation:
            for state, time in zip(states, times):
                animation.update(state, time=time)
    """
    return ConstraintPlotter(
        functions,
        level_sets,
        subtract_function,
        min_N=min_N,
        contour_kwargs=contour_kwargs,
        **plot_kwargs,
    )


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

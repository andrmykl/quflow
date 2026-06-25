import numpy as np
import pytest
import scipy.sparse as sp

from quflow.constraints import (
    BoundaryConditionPoisson,
    CommutatorKernelProjection,
    NonzeroEigenprojectorComplementProjection,
    level_set_eigenvects,
    project_to_eigenprojector_complement,
    project_to_commutator_kernel,
    project_to_trace_free_level_set,
    select_independent_rows_by_qr,
)


def test_real_interpolative_operator_is_not_supported():
    F_c = np.diag([1j, -1j])

    with pytest.raises(ValueError, match="Unknown row selection"):
        BoundaryConditionPoisson(
            F_c, N=2, row_selection="real_interpolative_operator"
        )


def test_row_norm_tolerance_drops_tiny_rows_before_normalization():
    C = sp.csr_matrix([[1e-13, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="above"):
        select_independent_rows_by_qr(C, rank=2, row_norm_tol=1e-12)

    V = select_independent_rows_by_qr(C, rank=2, row_norm_tol=0.0)

    assert V.shape == (2, 2)


def _retained_level_set_eigenvalues(F_c):
    return np.diag(-1j * F_c).real


def test_level_set_eigenvects_integer_keeps_closest_eigenvalues():
    eigenvalues = np.array([-5.0, -2.0, -1.0, 1.0, 4.0, 6.0])
    F = 1j * np.diag(eigenvalues)

    F_c, K = level_set_eigenvects(F, center=0.0, num_of_eigenvects=2)

    expected = np.array([0.0, 0.0, -1.0, 1.0, 0.0, 0.0])
    assert K == 2
    np.testing.assert_allclose(_retained_level_set_eigenvalues(F_c), expected)


def test_level_set_eigenvects_two_sided_counts():
    eigenvalues = np.array([-5.0, -2.0, -1.0, 1.0, 4.0, 6.0])
    F = 1j * np.diag(eigenvalues)

    F_c, K = level_set_eigenvects(F, center=0.0, num_of_eigenvects=[-2, 3])

    expected = np.array([0.0, -2.0, -1.0, 1.0, 4.0, 6.0])
    assert K == 5
    np.testing.assert_allclose(_retained_level_set_eigenvalues(F_c), expected)


def test_level_set_eigenvects_two_sided_infinity():
    eigenvalues = np.array([-5.0, -2.0, -1.0, 1.0, 4.0, 6.0])
    F = 1j * np.diag(eigenvalues)

    F_c, K = level_set_eigenvects(
        F, center=0.0, num_of_eigenvects=[-1, np.inf]
    )

    expected = np.array([0.0, 0.0, -1.0, 1.0, 4.0, 6.0])
    assert K == 4
    np.testing.assert_allclose(_retained_level_set_eigenvalues(F_c), expected)

    F_c, K = level_set_eigenvects(
        F, center=0.0, num_of_eigenvects=[-np.inf, 1]
    )

    expected = np.array([-5.0, -2.0, -1.0, 1.0, 0.0, 0.0])
    assert K == 4
    np.testing.assert_allclose(_retained_level_set_eigenvalues(F_c), expected)


def test_level_set_eigenvects_below_side_rank_range():
    eigenvalues = np.array([-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 1.0])
    F = 1j * np.diag(eigenvalues)

    F_c, K = level_set_eigenvects(F, center=0.0, num_of_eigenvects=[-5, -2])

    expected = np.array([0.0, -5.0, -4.0, -3.0, -2.0, 0.0, 0.0])
    assert K == 4
    np.testing.assert_allclose(_retained_level_set_eigenvalues(F_c), expected)


def test_level_set_eigenvects_above_side_rank_range():
    eigenvalues = np.array([-1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    F = 1j * np.diag(eigenvalues)

    F_c, K = level_set_eigenvects(F, center=0.0, num_of_eigenvects=[2, 5])

    expected = np.array([0.0, 0.0, 2.0, 3.0, 4.0, 5.0, 0.0])
    assert K == 4
    np.testing.assert_allclose(_retained_level_set_eigenvalues(F_c), expected)


def test_level_set_eigenvects_same_side_rank_range_with_infinity():
    eigenvalues = np.array([-1.0, 1.0, 2.0, 3.0, 4.0])
    F = 1j * np.diag(eigenvalues)

    F_c, K = level_set_eigenvects(F, center=0.0, num_of_eigenvects=[2, np.inf])

    expected = np.array([0.0, 0.0, 2.0, 3.0, 4.0])
    assert K == 3
    np.testing.assert_allclose(_retained_level_set_eigenvalues(F_c), expected)


def test_level_set_eigenvects_can_set_uniform_eigenvalue():
    eigenvalues = np.array([-3.0, -1.0, 1.0, 4.0])
    F = 1j * np.diag(eigenvalues)

    F_c, K = level_set_eigenvects(
        F, center=0.0, num_of_eigenvects=2, uniform_eigenvalue=5
    )

    expected = np.array([0.0, 5j, 5j, 0.0], dtype=complex)
    assert K == 2
    np.testing.assert_allclose(np.diag(F_c), expected)


def test_level_set_eigenvects_uniform_eigenvalue_conflicts_with_linear():
    eigenvalues = np.array([-1.0, 1.0])
    F = 1j * np.diag(eigenvalues)

    with pytest.raises(ValueError, match="linear=True"):
        level_set_eigenvects(
            F,
            center=0.0,
            num_of_eigenvects=1,
            linear=True,
            uniform_eigenvalue=5,
        )


def test_level_set_eigenvects_raises_when_side_count_is_unavailable():
    eigenvalues = np.array([-1.0, 2.0])
    F = 1j * np.diag(eigenvalues)

    with pytest.raises(ValueError, match="below center"):
        level_set_eigenvects(F, center=0.0, num_of_eigenvects=[-2, 1])


def test_project_to_trace_free_level_set_removes_trace_on_retained_subspace():
    eigenvalues = np.array([-3.0, -2.0, -1.0, 1.0])
    F = 1j * np.diag(eigenvalues)
    W = np.arange(16, dtype=float).reshape(4, 4).astype(complex)

    projected = project_to_trace_free_level_set(
        W, F, center=0.0, num_of_eigenvects=[-np.inf, -2]
    )

    expected = np.zeros_like(W)
    expected[:2, :2] = W[:2, :2]
    expected[:2, :2] -= np.trace(W[:2, :2]) / 2.0 * np.eye(2)

    np.testing.assert_allclose(projected, expected, atol=1e-12)
    np.testing.assert_allclose(np.trace(projected), 0.0, atol=1e-12)


def test_project_to_trace_free_level_set_preserves_skew_hermitian_batches():
    rng = np.random.default_rng(789)
    eigenvalues = np.array([-3.0, -2.0, -1.0, 1.0])
    F = 1j * np.diag(eigenvalues)
    A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    W = A - A.conj().T
    batch = np.stack([W, 2.0 * W])

    projected = project_to_trace_free_level_set(
        batch, F, center=0.0, num_of_eigenvects=[-np.inf, -2]
    )

    assert projected.shape == batch.shape
    np.testing.assert_allclose(
        projected + projected.conj().swapaxes(-1, -2), 0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.trace(projected, axis1=-2, axis2=-1), 0.0, atol=1e-12
    )


def test_commutator_kernel_projection_keeps_equal_eigenvalue_blocks():
    F_c = 1j * np.diag([1.0, 1.0, 2.0])
    X = np.array(
        [
            [1.0, 2.0 + 1.0j, 3.0],
            [4.0 - 2.0j, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=complex,
    )

    projector = CommutatorKernelProjection(F_c)
    projected = projector(X)

    expected = X.copy()
    expected[:2, 2] = 0.0
    expected[2, :2] = 0.0

    np.testing.assert_allclose(projected, expected)
    np.testing.assert_allclose(F_c @ projected - projected @ F_c, 0.0, atol=1e-12)
    np.testing.assert_allclose(projector(projected), projected, atol=1e-12)
    assert projector.kernel_dimension == 5
    assert projector.rank == 4


def test_project_to_commutator_kernel_preserves_skew_hermitian_batches():
    rng = np.random.default_rng(123)
    F_c = 1j * np.diag([0.0, 0.0, 3.0])
    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    W = A - A.conj().T
    batch = np.stack([W, 2.0 * W])

    projected = project_to_commutator_kernel(batch, F_c)

    assert projected.shape == batch.shape
    np.testing.assert_allclose(
        projected + projected.conj().swapaxes(-1, -2), 0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        F_c @ projected - projected @ F_c, 0.0, atol=1e-12
    )


def test_commutator_kernel_projection_linear_operator_matches_callable():
    F_c = 1j * np.diag([1.0, 2.0, 2.0])
    X = np.arange(9, dtype=float).reshape(3, 3).astype(complex)
    projector = CommutatorKernelProjection(F_c)

    op = projector.aslinearoperator()
    projected_vector = op @ X.ravel()

    np.testing.assert_allclose(
        projected_vector.reshape(3, 3), projector(X), atol=1e-12
    )


def test_nonzero_eigenprojector_complement_zeros_selected_components():
    F_c = 1j * np.diag([0.0, 2.0, 0.0, -3.0])
    X = np.arange(16, dtype=float).reshape(4, 4).astype(complex)

    projector = NonzeroEigenprojectorComplementProjection(F_c)
    projected = projector(X)

    expected = X.copy()
    expected[1, 1] = 0.0
    expected[3, 3] = 0.0

    assert projector.num_projectors == 2
    assert projector.kernel_dimension == 14
    np.testing.assert_allclose(projected, expected, atol=1e-12)
    np.testing.assert_allclose(projector.component_values(projected), 0.0, atol=1e-12)
    np.testing.assert_allclose(projector(projected), projected, atol=1e-12)


def test_nonzero_eigenprojector_complement_handles_rotated_eigenbasis():
    theta = 0.37
    Q = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=complex,
    )
    eigenvalues = np.array([0.0, 4.0, -2.0])
    F_c = Q @ (1j * np.diag(eigenvalues)) @ Q.conj().T
    X = np.array(
        [
            [1.0, 2.0 + 1.0j, 3.0],
            [4.0 - 2.0j, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=complex,
    )

    projected = project_to_eigenprojector_complement(X, F_c)
    projector = NonzeroEigenprojectorComplementProjection(F_c)

    np.testing.assert_allclose(projector.component_values(projected), 0.0, atol=1e-12)
    np.testing.assert_allclose(projector(projected), projected, atol=1e-12)


def test_nonzero_eigenprojector_complement_preserves_skew_hermitian_batches():
    rng = np.random.default_rng(456)
    F_c = 1j * np.diag([0.0, 1.0, -2.0])
    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    W = A - A.conj().T
    batch = np.stack([W, -0.5 * W])

    projected = project_to_eigenprojector_complement(batch, F_c)
    projector = NonzeroEigenprojectorComplementProjection(F_c)

    assert projected.shape == batch.shape
    np.testing.assert_allclose(
        projected + projected.conj().swapaxes(-1, -2), 0.0, atol=1e-12
    )
    np.testing.assert_allclose(projector.component_values(projected), 0.0, atol=1e-12)


def test_nonzero_eigenprojector_complement_linear_operator_matches_callable():
    F_c = 1j * np.diag([0.0, 1.0, 0.0])
    X = np.arange(9, dtype=float).reshape(3, 3).astype(complex)
    projector = NonzeroEigenprojectorComplementProjection(F_c)

    op = projector.aslinearoperator()
    projected_vector = op @ X.ravel()

    np.testing.assert_allclose(
        projected_vector.reshape(3, 3), projector(X), atol=1e-12
    )

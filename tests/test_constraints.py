import numpy as np

from quflow.constraints import BoundaryConditionPoisson


def _random_skew_hermitian_matrix(N, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    return X - X.conj().T


def _random_skew_hermitian_operator(N, seed):
    rng = np.random.default_rng(seed)
    H = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    H = (H + H.conj().T) / 2
    return 1j * H


def test_real_interpolative_operator_enforces_constraints():
    N = 4
    F_c = _random_skew_hermitian_operator(N, seed=1)
    W = _random_skew_hermitian_matrix(N, seed=2)

    solver = BoundaryConditionPoisson(
        F_c, N=N, row_selection="real_interpolative_operator"
    )
    P = solver.solve(W)

    assert solver.V.shape == (solver.constraint_rank, 2 * N**2)
    assert solver.boundary_residual(P) < 1e-10
    assert np.linalg.norm(F_c @ P - P @ F_c) < 1e-10
    assert np.linalg.norm(P.conj().T + P) < 1e-10
    assert abs(np.trace(P)) < 1e-10

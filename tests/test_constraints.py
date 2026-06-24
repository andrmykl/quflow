import numpy as np
import pytest
import scipy.sparse as sp

from quflow.constraints import BoundaryConditionPoisson, select_independent_rows_by_qr


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

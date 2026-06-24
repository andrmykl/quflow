import numpy as np
import pytest

from quflow.constraints import BoundaryConditionPoisson


def test_real_interpolative_operator_is_not_supported():
    F_c = np.diag([1j, -1j])

    with pytest.raises(ValueError, match="Unknown row selection"):
        BoundaryConditionPoisson(
            F_c, N=2, row_selection="real_interpolative_operator"
        )

import numpy as np

from code_construction.bicycle_code import build_bicycle_code


def test_bicycle_commutation():
    code = build_bicycle_code(m=8, first_row_support=[0, 1, 3])
    assert np.all((code.Hx @ code.Hz.T) % 2 == 0)


def test_bicycle_dimensions():
    m = 10
    code = build_bicycle_code(m=m, first_row_support=[0, 2, 5])
    assert code.C.shape == (m, m)
    assert code.Hx.shape == (m, 2 * m)
    assert code.Hz.shape == (m, 2 * m)
import numpy as np

from core.syndrome import compute_css_syndrome


def test_css_syndrome_shapes():
    Hx = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.uint8)
    Hz = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    ex = np.array([1, 0, 0, 1], dtype=np.uint8)
    ez = np.array([0, 1, 1, 0], dtype=np.uint8)

    sX, sZ = compute_css_syndrome(Hx, Hz, ex, ez)
    assert sX.shape == (2,)
    assert sZ.shape == (1,)


def test_css_syndrome_values():
    Hx = np.array([[1, 0, 1, 0]], dtype=np.uint8)
    Hz = np.array([[0, 1, 0, 1]], dtype=np.uint8)
    ex = np.array([0, 1, 0, 1], dtype=np.uint8)
    ez = np.array([1, 0, 1, 0], dtype=np.uint8)

    sX, sZ = compute_css_syndrome(Hx, Hz, ex, ez)
    assert np.array_equal(sX, np.array([0], dtype=np.uint8))
    assert np.array_equal(sZ, np.array([0], dtype=np.uint8))
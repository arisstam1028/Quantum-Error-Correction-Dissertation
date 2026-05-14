import numpy as np

from decoder.bp_decoder import BinaryBPDecoder


def test_bp_zero_syndrome():
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    decoder = BinaryBPDecoder(H, max_iters=10)
    syndrome = np.array([0, 0], dtype=np.uint8)

    result = decoder.decode(syndrome, p_error=0.05)
    assert result.estimated_error.shape == (3,)
    assert result.success


def test_bp_single_check():
    H = np.array([[1, 0, 0]], dtype=np.uint8)
    decoder = BinaryBPDecoder(H, max_iters=10)
    syndrome = np.array([1], dtype=np.uint8)

    result = decoder.decode(syndrome, p_error=0.05)
    assert result.success
    assert result.estimated_error[0] == 1
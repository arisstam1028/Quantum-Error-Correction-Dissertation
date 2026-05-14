import numpy as np

from core.helpers import GF2RowSpaceChecker, in_rowspace


def test_rowspace_checker_matches_in_rowspace():
    H = np.array(
        [
            [1, 0, 1, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 1],
        ],
        dtype=np.uint8,
    )
    checker = GF2RowSpaceChecker(H)

    vectors = [
        np.array([0, 0, 0, 0], dtype=np.uint8),
        np.array([1, 0, 1, 1], dtype=np.uint8),
        np.array([1, 1, 0, 1], dtype=np.uint8),
        np.array([1, 1, 1, 1], dtype=np.uint8),
    ]

    for v in vectors:
        assert checker.contains(v) == in_rowspace(v, H)

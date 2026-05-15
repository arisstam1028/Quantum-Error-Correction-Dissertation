import numpy as np


class CSSConverter:
    """
    Convert CSS parity-check matrices (Hx, Hz) into stabilizer matrix Hq in [X|Z] form:

        Hq  [[Hx, 0],
              [ 0, Hz]]

    All arithmetic is over GF(2).
    """

    @staticmethod
    def _to_gf2_matrix(M, *, name: str) -> np.ndarray:
        A = np.asarray(M, dtype=np.uint8)
        if A.ndim != 2:
            raise ValueError(f"{name} must be a 2D array-like.")
        return A & 1

    @staticmethod
    def validate_css(Hx, Hz) -> None:
        """
        Check CSS commutation: Hx * Hz^T  0 (mod 2).
        Raises ValueError if violated.
        """
        Hx = CSSConverter._to_gf2_matrix(Hx, name="Hx")
        Hz = CSSConverter._to_gf2_matrix(Hz, name="Hz")
        if Hx.shape[1] != Hz.shape[1]:
            raise ValueError(f"Hx and Hz must have same n columns. Got {Hx.shape[1]} and {Hz.shape[1]}.")

        # Compute Hx Hz^T over GF(2)
        prod = (Hx @ Hz.T) & 1
        if np.any(prod):
            # show positions of violations for debugging
            bad = np.argwhere(prod == 1)
            # keep message short but informative
            raise ValueError(
                "CSS condition violated: Hx @ Hz.T != 0 (mod 2). "
                f"First few violating entries (rowHx,rowHz): {bad[:10].tolist()}"
            )

    @staticmethod
    def css_to_Hq(Hx, Hz, *, validate: bool = True) -> np.ndarray:
        """
        Convert (Hx, Hz) to Hq in [X|Z] form.
        """
        Hx = CSSConverter._to_gf2_matrix(Hx, name="Hx")
        Hz = CSSConverter._to_gf2_matrix(Hz, name="Hz")

        if Hx.shape[1] != Hz.shape[1]:
            raise ValueError(f"Hx and Hz must have same n columns. Got {Hx.shape[1]} and {Hz.shape[1]}.")

        if validate:
            CSSConverter.validate_css(Hx, Hz)

        n = Hx.shape[1]
        top = np.hstack([Hx, np.zeros((Hx.shape[0], n), dtype=np.uint8)])
        bot = np.hstack([np.zeros((Hz.shape[0], n), dtype=np.uint8), Hz])
        return (np.vstack([top, bot]) & 1).astype(np.uint8)

    @staticmethod
    def infer_n_from_css(Hx, Hz) -> int:
        Hx = CSSConverter._to_gf2_matrix(Hx, name="Hx")
        Hz = CSSConverter._to_gf2_matrix(Hz, name="Hz")
        if Hx.shape[1] != Hz.shape[1]:
            raise ValueError("Hx and Hz must have same number of columns to infer n.")
        return int(Hx.shape[1])
# stabilizer_config.py
from __future__ import annotations

from Encoding_Circuit_Builder.main import build_five_qubit_encoder_bundle
from Table_Decoding_and_Error_Correction.stabilizer_measurement import StabilizerParser


PAPER_FIVE_QUBIT_STABILIZERS = [
    "XZZXI",
    "IXZZX",
    "XIXZZ",
    "ZXIXZ",
]


def get_hs_derived_stabilizers() -> list[str]:
    """
    Stabilizers generated from the standard-form Hs used by the encoder code.
    This is the current internal basis used by your existing code.
    """
    bundle = build_five_qubit_encoder_bundle(name="five_qubit_encoder")
    Hs = bundle["Hs"]
    return StabilizerParser.from_symplectic_rows(Hs)


def get_active_stabilizers(use_paper_stabilizers: bool = False) -> list[str]:
    """
    Returns the stabilizer basis to use for syndrome measurement / decoder.

    use_paper_stabilizers = False:
        Use current Hs-derived basis (existing behavior)

    use_paper_stabilizers = True:
        Use the paper's original 5-qubit cyclic stabilizers
    """
    if use_paper_stabilizers:
        return PAPER_FIVE_QUBIT_STABILIZERS.copy()
    return get_hs_derived_stabilizers()
import numpy as np
from stabilizer_logicals import StabilizerPipeline, GF2
#from stabilizer_logicals_v2 import StabilizerPipeline, GF2
#from stabilizer_logicals_v3 import StabilizerPipeline, GF2
#from stabilizer_logicals_v4 import StabilizerPipeline, GF2
from css_converter import CSSConverter
from css_logicals import CSSLogicalOperatorCalculator

def demo_with_stabilizers():
    stabilizers = [
        "XZZXI",
        "IXZZX",
        "XIXZZ",
        "ZXIXZ",
    ]

    # k inferred automatically
    StabilizerPipeline.run(stabilizers=stabilizers, k=None)
    print(' Demo: input stabilizers ')


def demo_with_Hq():
    Hq = GF2.as_u8(np.array([
        [1,0,0,1,0,  0,1,1,0,0],
        [0,1,0,0,1,  0,0,1,1,0],
        [1,0,1,0,0,  0,0,0,1,1],
        [0,1,0,1,0,  1,0,0,0,1],
    ], dtype=np.uint8))

    # k inferred automatically
    StabilizerPipeline.run(Hq=Hq, k=None)
    print('\n Demo: input Hq ')

def demo_css_steane():
    Hx = np.array([
        [1, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 1],
    ], dtype=np.uint8)

    Hz = Hx.copy()  # < FIX: define Hz

    n = Hx.shape[1]

    # Build Hq  [Hx | 0 ; 0 | Hz]
    Hq = np.zeros((Hx.shape[0] + Hz.shape[0], 2 * n), dtype=np.uint8)
    Hq[:Hx.shape[0], :n] = Hx
    Hq[Hx.shape[0]:, n:] = Hz

    StabilizerPipeline.run(Hq=Hq)

    #  NEW: compute CSS logical operators robustly (kernel/quotient method) 
    # This avoids relying on the Eq.(19)-(20) "Hs standard form" construction,
    # which is not guaranteed by the current StandardFormBuilder for all CSS codes.
    # res  CSSLogicalOperatorCalculator.compute_from_Hq_css(Hq, reduce_dependentTrue)
    #
    # print("\n CSS logical operators (computed from Hx/Hz) ")
    # print(f"n  {res.n}, k  {res.k}")
    # print("Hx:")
    # for row in res.Hx:
    #     print(" ", "".join(str(int(b)) for b in row.tolist()))
    # print("Hz:")
    # for row in res.Hz:
    #     print(" ", "".join(str(int(b)) for b in row.tolist()))
    #
    # for i in range(res.k):
    #     x  res.Xbars[i][:res.n]
    #     z  res.Xbars[i][res.n:]
    #     print(f"  Xbar[{i}]  {''.join(str(int(b)) for b in x)} | {''.join(str(int(b)) for b in z)}")
    #
    #     x  res.Zbars[i][:res.n]
    #     z  res.Zbars[i][res.n:]
    #     print(f"  Zbar[{i}]  {''.join(str(int(b)) for b in x)} | {''.join(str(int(b)) for b in z)}")

def main():
    #demo_with_stabilizers()
    demo_with_Hq()
    #demo_css_steane()

if __name__ == "__main__":
    main()
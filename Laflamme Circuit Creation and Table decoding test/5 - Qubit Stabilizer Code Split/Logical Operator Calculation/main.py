import numpy as np
from stabilizer_logicals import StabilizerPipeline, GF2

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

def main():
    #demo_with_stabilizers()
    demo_with_Hq()

if __name__ == "__main__":
    main()
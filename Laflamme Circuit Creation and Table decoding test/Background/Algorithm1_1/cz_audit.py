# cz_audit.py
from __future__ import annotations
from collections import Counter
from typing import List, Tuple

def expected_cz_edges(Hs: List[List[int]], logical_X: List[int]) -> Counter[Tuple[int,int,str]]:
    """
    Returns multiset of expected CZ edges as (min(a,b), max(a,b), source_tag).
    source_tag  'L' for logical step, 'S<i>' for stabilizer row i.
    """
    m = len(Hs)
    n = len(Hs[0]) // 2
    k = n - m
    msg = n - k  # for k1, msgn-1

    xL = logical_X[:n]
    zL = logical_X[n:]

    out = Counter()

    # Logical CZs
    for j in range(n):
        if j == msg:
            continue
        if (xL[j], zL[j]) == (0, 1):
            a, b = sorted((msg, j))
            out[(a, b, "L")] += 1

    # Stabilizer CZs (Algorithm 1 uses i0..r-1; if you use full rows, change range)
    # Here we assume you used ctrli for each row you looped over.
    for i in range(m):
        ctrl = i
        for j in range(n):
            if j == ctrl:
                continue
            xbit = Hs[i][j]
            zbit = Hs[i][n + j]
            if (xbit, zbit) == (0, 1):
                a, b = sorted((ctrl, j))
                out[(a, b, f"S{i}")] += 1

    return out

def actual_cz_edges(qc) -> Counter[Tuple[int,int]]:
    """
    Extract multiset of actual CZ edges as unordered pairs (min(a,b), max(a,b)).
    Note: Qiskit's CZ is symmetric, so we treat it as an unordered edge.
    """
    out = Counter()
    for inst, qargs, _ in qc.data:
        if inst.name == "cz":
            a = qargs[0]._index
            b = qargs[1]._index
            a, b = sorted((a, b))
            out[(a, b)] += 1
    return out

def audit_cz(Hs: List[List[int]], logical_X: List[int], qc) -> None:
    exp = expected_cz_edges(Hs, logical_X)
    act = actual_cz_edges(qc)

    # Collapse expected across tags for comparison with circuit counts
    exp_pairs = Counter()
    for (a, b, tag), c in exp.items():
        exp_pairs[(a, b)] += c

    print("Expected CZ pair counts:")
    for k in sorted(exp_pairs):
        print(f"  CZ {k}: {exp_pairs[k]}")
    print("\nActual CZ pair counts:")
    for k in sorted(act):
        print(f"  CZ {k}: {act[k]}")

    # Differences
    all_pairs = set(exp_pairs) | set(act)
    extra = {p: act[p] - exp_pairs[p] for p in all_pairs if act[p] > exp_pairs[p]}
    missing = {p: exp_pairs[p] - act[p] for p in all_pairs if exp_pairs[p] > act[p]}

    print("\nDifferences:")
    if not extra and not missing:
        print("  ✅ CZ pairs match exactly (as a multiset).")
    else:
        if extra:
            print("  Extra CZ in circuit:")
            for p in sorted(extra):
                print(f"    {p}: +{extra[p]}")
        if missing:
            print("  Missing CZ in circuit:")
            for p in sorted(missing):
                print(f"    {p}: -{missing[p]}")

    # Attribution: where each expected CZ came from
    print("\nAttribution of expected CZs (logical vs stabilizer rows):")
    by_pair = {}
    for (a, b, tag), c in exp.items():
        by_pair.setdefault((a, b), []).append((tag, c))
    for pair in sorted(by_pair):
        items = ", ".join([f"{tag}×{c}" for tag, c in sorted(by_pair[pair])])
        print(f"  CZ {pair}: {items}")
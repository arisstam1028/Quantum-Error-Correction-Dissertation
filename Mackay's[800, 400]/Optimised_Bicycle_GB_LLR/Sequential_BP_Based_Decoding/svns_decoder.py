from dataclasses import dataclass

import numpy as np

from decoder.bp_graph import BPGraph, build_bp_graph
from decoder.decoder_result import DecoderResult


def _prefix_suffix_products(values: np.ndarray) -> np.ndarray:
    d = values.size
    if d == 0:
        return np.empty(0, dtype=np.float64)
    if d == 1:
        return np.ones(1, dtype=np.float64)

    prefix = np.ones(d, dtype=np.float64)
    suffix = np.ones(d, dtype=np.float64)

    for i in range(1, d):
        prefix[i] = prefix[i - 1] * values[i - 1]
    for i in range(d - 2, -1, -1):
        suffix[i] = suffix[i + 1] * values[i + 1]

    return prefix * suffix


def _parity_from_check_edges(estimate: np.ndarray, edge_var: np.ndarray, check_edge_ptr: np.ndarray) -> np.ndarray:
    predicted = np.zeros(check_edge_ptr.size - 1, dtype=np.uint8)

    for c in range(predicted.size):
        start = int(check_edge_ptr[c])
        end = int(check_edge_ptr[c + 1])
        if start == end:
            continue
        predicted[c] = np.bitwise_xor.reduce(estimate[edge_var[start:end]])

    return predicted


@dataclass
class SVNSBPDecoder:
    H: np.ndarray
    max_iters: int = 30
    epsilon: float = 1e-12
    llr_clip: float = 50.0

    def __post_init__(self) -> None:
        self.H = np.asarray(self.H, dtype=np.uint8)
        self.graph: BPGraph = build_bp_graph(self.H)
        self.m, self.n = self.H.shape
        self.num_edges = int(self.graph.edge_var.size)
        self.vn_order = list(range(self.n))
        self.edge_var = self.graph.edge_var
        self.edge_check = self.graph.edge_check
        self.check_edge_ptr = self.graph.check_edge_ptr
        self.var_edge_ptr = self.graph.var_edge_ptr
        self.var_edges = self.graph.var_edges
        self.edge_pos_in_check = self.graph.edge_pos_in_check

    def _channel_llr(self, p_error: float) -> float:
        p = float(np.clip(p_error, self.epsilon, 1.0 - self.epsilon))
        return float(np.log((1.0 - p) / p))

    def decode(self, syndrome: np.ndarray, p_error: float) -> DecoderResult:
        """
        Sequential Variable Node Scheduling (SVNS) in LLR domain.

        Matches the paper structure:
        - for each variable v in order:
            1) refresh all incoming mc->v
            2) bitLLR_v = mu + sum incoming
            3) update all outgoing mv->c
        """
        syndrome = np.asarray(syndrome, dtype=np.uint8)
        if syndrome.shape != (self.m,):
            raise ValueError(f"syndrome must have shape ({self.m},)")

        mu = self._channel_llr(p_error)
        sigma = np.where(syndrome == 1, -1.0, 1.0).astype(np.float64)

        mv_to_c = np.full(self.num_edges, mu, dtype=np.float64)
        mc_to_v = np.zeros(self.num_edges, dtype=np.float64)
        bit_llr = np.full(self.n, mu, dtype=np.float64)
        estimate = np.zeros(self.n, dtype=np.uint8)

        converged = False
        iters_used = 0

        for it in range(1, self.max_iters + 1):
            for v in self.vn_order:
                vstart = int(self.var_edge_ptr[v])
                vend = int(self.var_edge_ptr[v + 1])
                v_edges = self.var_edges[vstart:vend]
                if v_edges.size == 0:
                    bit_llr[v] = mu
                    continue

                # Refresh incoming mc->v for all incident checks
                tanh_half_mv = np.tanh(0.5 * np.clip(mv_to_c, -self.llr_clip, self.llr_clip))
                for e in v_edges:
                    c = int(self.edge_check[e])
                    cstart = int(self.check_edge_ptr[c])
                    cend = int(self.check_edge_ptr[c + 1])

                    vals = tanh_half_mv[cstart:cend]
                    excl_prod = _prefix_suffix_products(vals)

                    local_pos = int(self.edge_pos_in_check[e])
                    arg = sigma[c] * excl_prod[local_pos]
                    arg = float(np.clip(arg, -1.0 + self.epsilon, 1.0 - self.epsilon))

                    mc_to_v[e] = 2.0 * np.arctanh(arg)
                    mc_to_v[e] = float(np.clip(mc_to_v[e], -self.llr_clip, self.llr_clip))

                # bitLLR_v
                bit_llr[v] = mu + float(np.sum(mc_to_v[v_edges], dtype=np.float64))
                bit_llr[v] = float(np.clip(bit_llr[v], -self.llr_clip, self.llr_clip))

                # Update all outgoing mv->c for this variable
                mv_to_c[v_edges] = np.clip(bit_llr[v] - mc_to_v[v_edges], -self.llr_clip, self.llr_clip)

            estimate = (bit_llr < 0.0).astype(np.uint8)
            predicted = _parity_from_check_edges(estimate, self.edge_var, self.check_edge_ptr)
            iters_used = it

            if np.array_equal(predicted, syndrome):
                converged = True
                break

        final_predicted = _parity_from_check_edges(estimate, self.edge_var, self.check_edge_ptr)
        residual = (syndrome ^ final_predicted).astype(np.uint8)
        success = bool(np.all(residual == 0))

        return DecoderResult(
            estimated_error=estimate.copy(),
            success=success,
            iterations_used=iters_used,
            residual_syndrome=residual,
            converged=converged,
        )

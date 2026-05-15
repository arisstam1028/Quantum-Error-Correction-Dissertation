"""
Purpose:
    Implement sequential variable-node scheduling for binary BP.

Process:
    Visit variables in a fixed order, refresh their incoming check messages,
    update the variable belief, and immediately update outgoing messages.

Theory link:
    Sequential scheduling can converge differently from flooding because
    later variables in an iteration see fresher local information.
"""

from dataclasses import dataclass

import numpy as np

from decoder.bp_graph import BPGraph, build_bp_graph
from decoder.decoder_result import DecoderResult


def _prefix_suffix_products_inplace(
    values: np.ndarray,
    prefix: np.ndarray,
    suffix: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    """
    Compute products excluding one recipient edge using caller-owned buffers.
    """
    d = values.size
    if d == 0:
        return out[:0]
    if d == 1:
        out[0] = 1.0
        return out[:1]

    prefix[0] = 1.0
    for i in range(1, d):
        prefix[i] = prefix[i - 1] * values[i - 1]

    suffix[d - 1] = 1.0
    for i in range(d - 2, -1, -1):
        suffix[i] = suffix[i + 1] * values[i + 1]

    np.multiply(prefix[:d], suffix[:d], out=out[:d])
    return out[:d]


def _parity_from_check_edges(
    estimate: np.ndarray,
    edge_var: np.ndarray,
    check_edge_ptr: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    """
    Compute the predicted syndrome for the current hard decision.
    """
    out.fill(0)

    for c in range(out.size):
        start = int(check_edge_ptr[c])
        end = int(check_edge_ptr[c + 1])
        if start == end:
            continue
        out[c] = np.sum(estimate[edge_var[start:end]], dtype=np.int32) & 1

    return out


@dataclass
class SVNSBPDecoder:
    H: np.ndarray
    max_iters: int = 30
    epsilon: float = 1e-12
    llr_clip: float = 50.0

    def __post_init__(self) -> None:
        """
        Build graph metadata and reusable buffers for SVNS updates.
        """
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
        self.max_check_degree = int(np.max(np.diff(self.check_edge_ptr), initial=0))

        self._mv_to_c = np.empty(self.num_edges, dtype=np.float64)
        self._mc_to_v = np.empty(self.num_edges, dtype=np.float64)
        self._bit_llr = np.empty(self.n, dtype=np.float64)
        self._estimate = np.empty(self.n, dtype=np.uint8)
        self._predicted = np.empty(self.m, dtype=np.uint8)
        self._residual = np.empty(self.m, dtype=np.uint8)
        self._clip_buf = np.empty(self.max_check_degree, dtype=np.float64)
        self._tanh_buf = np.empty(self.max_check_degree, dtype=np.float64)
        self._prefix_buf = np.empty(self.max_check_degree, dtype=np.float64)
        self._suffix_buf = np.empty(self.max_check_degree, dtype=np.float64)
        self._prod_buf = np.empty(self.max_check_degree, dtype=np.float64)
        self._arg_buf = np.empty(self.max_check_degree, dtype=np.float64)

    def _channel_llr(self, p_error: float) -> float:
        """
        Convert a binary error prior into an LLR.
        """
        p = float(np.clip(p_error, self.epsilon, 1.0 - self.epsilon))
        return float(np.log((1.0 - p) / p))

    def decode(self, syndrome: np.ndarray, p_error: float) -> DecoderResult:
        """
        Sequential Variable Node Scheduling (SVNS) in LLR domain.

        Matches the paper structure:
        - for each variable v in order:
            1) refresh all incoming mc->v
            2) bitLLR_v  mu + sum incoming
            3) update all outgoing mv->c
        """
        syndrome = np.asarray(syndrome, dtype=np.uint8)
        if syndrome.shape != (self.m,):
            raise ValueError(f"syndrome must have shape ({self.m},)")

        mu = self._channel_llr(p_error)
        sigma = np.where(syndrome == 1, -1.0, 1.0).astype(np.float64)

        mv_to_c = self._mv_to_c
        mv_to_c.fill(mu)
        mc_to_v = self._mc_to_v
        mc_to_v.fill(0.0)
        bit_llr = self._bit_llr
        bit_llr.fill(mu)
        estimate = self._estimate
        estimate.fill(0)
        predicted = self._predicted

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

                # Refresh incoming mc->v for all incident checks.
                for e in v_edges:
                    c = int(self.edge_check[e])
                    cstart = int(self.check_edge_ptr[c])
                    cend = int(self.check_edge_ptr[c + 1])
                    degree = cend - cstart

                    clip_vals = self._clip_buf[:degree]
                    np.clip(mv_to_c[cstart:cend], -self.llr_clip, self.llr_clip, out=clip_vals)
                    clip_vals *= 0.5
                    tanh_vals = self._tanh_buf[:degree]
                    np.tanh(clip_vals, out=tanh_vals)
                    excl_prod = _prefix_suffix_products_inplace(
                        tanh_vals,
                        self._prefix_buf,
                        self._suffix_buf,
                        self._prod_buf,
                    )

                    local_pos = int(self.edge_pos_in_check[e])
                    arg = float(np.clip(sigma[c] * excl_prod[local_pos], -1.0 + self.epsilon, 1.0 - self.epsilon))
                    mc_to_v[e] = float(np.clip(2.0 * np.arctanh(arg), -self.llr_clip, self.llr_clip))

                # Variable belief after refreshed incoming check messages.
                bit_llr[v] = mu + float(np.sum(mc_to_v[v_edges], dtype=np.float64))
                bit_llr[v] = float(np.clip(bit_llr[v], -self.llr_clip, self.llr_clip))

                # Update all outgoing variable-to-check messages for this variable.
                np.clip(bit_llr[v] - mc_to_v[v_edges], -self.llr_clip, self.llr_clip, out=mv_to_c[v_edges])

            np.less(bit_llr, 0.0, out=estimate)
            _parity_from_check_edges(estimate, self.edge_var, self.check_edge_ptr, predicted)
            iters_used = it

            if np.array_equal(predicted, syndrome):
                converged = True
                break

        _parity_from_check_edges(estimate, self.edge_var, self.check_edge_ptr, predicted)
        np.bitwise_xor(syndrome, predicted, out=self._residual)
        residual = self._residual
        success = bool(np.all(residual == 0))

        return DecoderResult(
            estimated_error=estimate.copy(),
            success=success,
            iterations_used=iters_used,
            residual_syndrome=residual,
            converged=converged,
        )

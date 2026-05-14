"""
Purpose:
    Implement sequential check-node scheduling for binary BP.

Process:
    Visit checks in a fixed order, refresh that check's outgoing messages,
    update affected variable beliefs, and propagate new variable messages.

Theory link:
    SCNS uses fresher check information within an iteration, which can change
    convergence behavior compared with flooding BP and SVNS.
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
class SCNSBPDecoder:
    H: np.ndarray
    max_iters: int = 30
    epsilon: float = 1e-12
    llr_clip: float = 50.0

    def __post_init__(self) -> None:
        """
        Build graph metadata and reusable buffers for SCNS updates.
        """
        self.H = np.asarray(self.H, dtype=np.uint8)
        self.graph: BPGraph = build_bp_graph(self.H)
        self.m, self.n = self.H.shape
        self.num_edges = int(self.graph.edge_var.size)
        self.cn_order = list(range(self.m))
        self.edge_var = self.graph.edge_var
        self.check_edge_ptr = self.graph.check_edge_ptr
        self.var_edge_ptr = self.graph.var_edge_ptr
        self.var_edges = self.graph.var_edges
        self.max_check_degree = int(np.max(np.diff(self.check_edge_ptr), initial=0))

        self._mv_to_c = np.empty(self.num_edges, dtype=np.float64)
        self._mc_to_v = np.empty(self.num_edges, dtype=np.float64)
        self._cached_check_msgs = np.empty(self.num_edges, dtype=np.float64)
        self._bit_llr = np.empty(self.n, dtype=np.float64)
        self._estimate = np.empty(self.n, dtype=np.uint8)
        self._predicted = np.empty(self.m, dtype=np.uint8)
        self._residual = np.empty(self.m, dtype=np.uint8)
        self._clip_buf = np.empty(self.max_check_degree, dtype=np.float64)
        self._tanh_buf = np.empty(self.max_check_degree, dtype=np.float64)
        self._prefix_buf = np.empty(self.max_check_degree, dtype=np.float64)
        self._suffix_buf = np.empty(self.max_check_degree, dtype=np.float64)
        self._prod_buf = np.empty(self.max_check_degree, dtype=np.float64)
        self._var_sum = np.empty(self.n, dtype=np.float64)

    def _refresh_check_messages(
        self,
        mv_to_c: np.ndarray,
        sigma: np.ndarray,
        check_messages: np.ndarray,
        c: int,
    ) -> None:
        """
        Recompute all outgoing messages for one check node.

        Role in pipeline:
            Gives SCNS a fresh local check update before the connected
            variable messages are refreshed.
        """
        cstart = int(self.check_edge_ptr[c])
        cend = int(self.check_edge_ptr[c + 1])
        if cstart == cend:
            return

        degree = cend - cstart
        clip_vals = self._clip_buf[:degree]
        np.clip(mv_to_c[cstart:cend], -self.llr_clip, self.llr_clip, out=clip_vals)
        clip_vals *= 0.5
        tanh_half = self._tanh_buf[:degree]
        np.tanh(clip_vals, out=tanh_half)
        excl_prod = _prefix_suffix_products_inplace(
            tanh_half,
            self._prefix_buf,
            self._suffix_buf,
            self._prod_buf,
        )
        np.multiply(excl_prod, sigma[c], out=check_messages[cstart:cend])
        np.clip(check_messages[cstart:cend], -1.0 + self.epsilon, 1.0 - self.epsilon, out=check_messages[cstart:cend])
        np.arctanh(check_messages[cstart:cend], out=check_messages[cstart:cend])
        check_messages[cstart:cend] *= 2.0
        np.clip(check_messages[cstart:cend], -self.llr_clip, self.llr_clip, out=check_messages[cstart:cend])

    def _channel_llr(self, p_error: float) -> float:
        """
        Convert a binary error prior into an LLR.
        """
        p = float(np.clip(p_error, self.epsilon, 1.0 - self.epsilon))
        return float(np.log((1.0 - p) / p))

    def decode(self, syndrome: np.ndarray, p_error: float) -> DecoderResult:
        """
        Sequential Check Node Scheduling (SCNS) in LLR domain.

        Preserved from the paper:
        - fixed check order
        - refresh current check messages first
        - for each (v, c), update only mv->c
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
        cached_check_msgs = self._cached_check_msgs
        cached_check_msgs.fill(0.0)

        for c in self.cn_order:
            self._refresh_check_messages(mv_to_c, sigma, cached_check_msgs, c)

        bit_llr = self._bit_llr
        bit_llr.fill(mu)
        estimate = self._estimate
        estimate.fill(0)
        predicted = self._predicted

        converged = False
        iters_used = 0

        for it in range(1, self.max_iters + 1):
            self._var_sum.fill(0.0)
            np.add.at(self._var_sum, self.edge_var, cached_check_msgs)

            for c in self.cn_order:
                cstart = int(self.check_edge_ptr[c])
                cend = int(self.check_edge_ptr[c + 1])
                if cstart == cend:
                    continue

                mc_to_v[cstart:cend] = cached_check_msgs[cstart:cend]

                for e in range(cstart, cend):
                    v = int(self.edge_var[e])
                    total = mu + self._var_sum[v] - cached_check_msgs[e] + mc_to_v[e]
                    bit_llr[v] = float(np.clip(total, -self.llr_clip, self.llr_clip))
                    mv_to_c[e] = float(np.clip(bit_llr[v] - mc_to_v[e], -self.llr_clip, self.llr_clip))

                self._refresh_check_messages(mv_to_c, sigma, cached_check_msgs, c)
                np.add.at(self._var_sum, self.edge_var[cstart:cend], cached_check_msgs[cstart:cend] - mc_to_v[cstart:cend])

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

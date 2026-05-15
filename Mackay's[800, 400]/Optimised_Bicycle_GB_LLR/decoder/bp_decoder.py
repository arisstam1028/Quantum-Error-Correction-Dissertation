"""
Purpose:
    Implement optimized flooding-schedule binary BP decoding.

Process:
    Initialize edge messages from the channel prior, update all check nodes,
    update all variable nodes, make a hard decision, and stop when the
    predicted syndrome matches the observed syndrome.

Theory link:
    This decoder solves one binary parity problem for a CSS component. The
    full quantum frame is decoded by running it once for X errors and once
    for Z errors.
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
    For values[0..d-1], return out[k]  product of all values except values[k].

    Role in pipeline:
        Computes the check-node exclusion product without allocating a new
        array for each edge update.
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
    Compute the syndrome predicted by a hard-decision estimate.

    Role in pipeline:
        Tests whether the current BP estimate satisfies the requested parity
        checks and can be accepted early.
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
class BinaryBPDecoder:
    H: np.ndarray
    max_iters: int = 30
    epsilon: float = 1e-12
    llr_clip: float = 50.0

    def __post_init__(self) -> None:
        """
        Build graph metadata and reusable work buffers.

        Role in pipeline:
            Keeps the optimized decoder allocation-light during long Monte
            Carlo runs.
        """
        self.H = np.asarray(self.H, dtype=np.uint8)
        self.graph: BPGraph = build_bp_graph(self.H)
        self.m, self.n = self.H.shape
        self.num_edges = int(self.graph.edge_var.size)
        self.edge_var = self.graph.edge_var
        self.check_edge_ptr = self.graph.check_edge_ptr
        self.max_check_degree = int(np.max(np.diff(self.check_edge_ptr), initial=0))

        self._mv_to_c = np.empty(self.num_edges, dtype=np.float64)
        self._mc_to_v = np.empty(self.num_edges, dtype=np.float64)
        self._bit_llr = np.empty(self.n, dtype=np.float64)
        self._estimate = np.empty(self.n, dtype=np.uint8)
        self._predicted = np.empty(self.m, dtype=np.uint8)
        self._residual = np.empty(self.m, dtype=np.uint8)
        self._clip_buffer = np.empty(self.num_edges, dtype=np.float64)
        self._tanh_half_mv = np.empty(self.num_edges, dtype=np.float64)
        self._prefix_buf = np.empty(self.max_check_degree, dtype=np.float64)
        self._suffix_buf = np.empty(self.max_check_degree, dtype=np.float64)
        self._prod_buf = np.empty(self.max_check_degree, dtype=np.float64)
        self._arg_buf = np.empty(self.max_check_degree, dtype=np.float64)

    def _channel_llr(self, p_error: float) -> float:
        """
        Convert a Bernoulli error probability into a log-likelihood ratio.
        """
        p = float(np.clip(p_error, self.epsilon, 1.0 - self.epsilon))
        return float(np.log((1.0 - p) / p))

    def decode(self, syndrome: np.ndarray, p_error: float) -> DecoderResult:
        """
        Flooding BP in LLR domain.

        Preserved:
        - same decode signature
        - same hard decision x_hat[v]  1 if bitLLR[v] < 0 else 0
        - same syndrome stopping rule
        """
        syndrome = np.asarray(syndrome, dtype=np.uint8)
        if syndrome.shape != (self.m,):
            raise ValueError(f"syndrome must have shape ({self.m},)")

        mu = self._channel_llr(p_error)
        sigma = np.where(syndrome == 1, -1.0, 1.0).astype(np.float64)

        # Message arrays for variable-to-check and check-to-variable beliefs.
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
            # Check update using the target syndrome sign.
            np.clip(mv_to_c, -self.llr_clip, self.llr_clip, out=self._clip_buffer)
            np.multiply(self._clip_buffer, 0.5, out=self._clip_buffer)
            np.tanh(self._clip_buffer, out=self._tanh_half_mv)

            for c in range(self.m):
                start = int(self.check_edge_ptr[c])
                end = int(self.check_edge_ptr[c + 1])
                if start == end:
                    continue

                vals = self._tanh_half_mv[start:end]
                degree = end - start
                excl_prod = _prefix_suffix_products_inplace(
                    vals,
                    self._prefix_buf,
                    self._suffix_buf,
                    self._prod_buf,
                )
                arg = self._arg_buf[:degree]
                np.multiply(excl_prod, sigma[c], out=arg)
                np.clip(arg, -1.0 + self.epsilon, 1.0 - self.epsilon, out=arg)

                np.arctanh(arg, out=mc_to_v[start:end])
                mc_to_v[start:end] *= 2.0
                np.clip(mc_to_v[start:end], -self.llr_clip, self.llr_clip, out=mc_to_v[start:end])

            # Variable update combines the channel prior with check messages.
            bit_llr.fill(mu)
            np.add.at(bit_llr, self.edge_var, mc_to_v)
            np.clip(bit_llr, -self.llr_clip, self.llr_clip, out=bit_llr)
            np.subtract(bit_llr[self.edge_var], mc_to_v, out=mv_to_c)
            np.clip(mv_to_c, -self.llr_clip, self.llr_clip, out=mv_to_c)

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

from dataclasses import dataclass

import numpy as np

from .bp_graph import BPGraph, build_bp_graph
from .decoder_result import DecoderResult


def _prefix_suffix_products(values: np.ndarray) -> np.ndarray:
    """
    For values[0..d-1], return out[k] = product of all values except values[k].
    """
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


def _parity_from_check_edges(
    estimate: np.ndarray,
    edge_var: np.ndarray,
    check_edge_ptr: np.ndarray,
) -> np.ndarray:
    predicted = np.zeros(check_edge_ptr.size - 1, dtype=np.uint8)

    for c in range(predicted.size):
        start = int(check_edge_ptr[c])
        end = int(check_edge_ptr[c + 1])
        if start == end:
            continue
        predicted[c] = np.bitwise_xor.reduce(estimate[edge_var[start:end]])

    return predicted


@dataclass
class BinaryBPDecoder:
    H: np.ndarray
    max_iters: int = 30
    epsilon: float = 1e-12
    llr_clip: float = 50.0

    def __post_init__(self) -> None:
        self.H = np.asarray(self.H, dtype=np.uint8)
        self.graph: BPGraph = build_bp_graph(self.H)
        self.m, self.n = self.H.shape
        self.num_edges = int(self.graph.edge_var.size)
        self.edge_var = self.graph.edge_var
        self.check_edge_ptr = self.graph.check_edge_ptr

    def _channel_llr(self, p_error: float) -> float:
        p = float(np.clip(p_error, self.epsilon, 1.0 - self.epsilon))
        return float(np.log((1.0 - p) / p))

    def decode(self, syndrome: np.ndarray, p_error: float) -> DecoderResult:
        syndrome = np.asarray(syndrome, dtype=np.uint8)
        if syndrome.shape != (self.m,):
            raise ValueError(f"syndrome must have shape ({self.m},)")

        mu = self._channel_llr(p_error)
        sigma = np.where(syndrome == 1, -1.0, 1.0).astype(np.float64)

        # messages
        mv_to_c = np.full(self.num_edges, mu, dtype=np.float64)
        mc_to_v = np.zeros(self.num_edges, dtype=np.float64)

        bit_llr = np.full(self.n, mu, dtype=np.float64)
        estimate = np.zeros(self.n, dtype=np.uint8)

        converged = False
        iters_used = 0

        for it in range(1, self.max_iters + 1):
            # Check update
            tanh_half_mv = np.tanh(0.5 * np.clip(mv_to_c, -self.llr_clip, self.llr_clip))

            for c in range(self.m):
                start = int(self.check_edge_ptr[c])
                end = int(self.check_edge_ptr[c + 1])
                if start == end:
                    continue

                vals = tanh_half_mv[start:end]
                excl_prod = _prefix_suffix_products(vals)
                arg = sigma[c] * excl_prod
                arg = np.clip(arg, -1.0 + self.epsilon, 1.0 - self.epsilon)

                mc_to_v[start:end] = 2.0 * np.arctanh(arg)
                mc_to_v[start:end] = np.clip(mc_to_v[start:end], -self.llr_clip, self.llr_clip)

            # Variable update
            bit_llr.fill(mu)
            np.add.at(bit_llr, self.edge_var, mc_to_v)
            np.clip(bit_llr, -self.llr_clip, self.llr_clip, out=bit_llr)

            mv_to_c[:] = bit_llr[self.edge_var] - mc_to_v
            np.clip(mv_to_c, -self.llr_clip, self.llr_clip, out=mv_to_c)

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
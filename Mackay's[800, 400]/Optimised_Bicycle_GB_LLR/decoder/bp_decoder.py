from dataclasses import dataclass

import numpy as np

from decoder.bp_graph import BPGraph, build_bp_graph
from decoder.decoder_result import DecoderResult


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

    # Random perturbation options
    enable_random_perturbation: bool = False
    perturb_iters: int = 40
    perturb_max_feedbacks: int = 40
    perturb_strength: float = 0.1
    random_seed: int | None = None

    def __post_init__(self) -> None:
        self.H = np.asarray(self.H, dtype=np.uint8)
        self.graph: BPGraph = build_bp_graph(self.H)
        self.m, self.n = self.H.shape
        self.num_edges = int(self.graph.edge_var.size)

        self.edge_var = self.graph.edge_var
        self.check_edge_ptr = self.graph.check_edge_ptr
        self.check_to_var = self.graph.check_to_var

        self.rng = np.random.default_rng(self.random_seed)

    def _channel_llr(self, p_error: float) -> float:
        p = float(np.clip(p_error, self.epsilon, 1.0 - self.epsilon))
        return float(np.log((1.0 - p) / p))

    def _base_prior_llr(self, p_error: float) -> np.ndarray:
        mu = self._channel_llr(p_error)
        return np.full(self.n, mu, dtype=np.float64)

    def _perturb_prior_llr(
        self,
        base_prior_llr: np.ndarray,
        p_error: float,
        frustrated_check: int,
    ) -> np.ndarray:
        """
        Binary analogue of random perturbation:
        for all variables attached to one randomly chosen frustrated check,
        scale P(1) by (1 + delta_v), renormalize, then convert back to LLR.

        With binary p = P(1), this yields:
            p'_v = ((1 + delta_v) * p) / (1 + delta_v * p)
            L'_v = log((1 - p'_v) / p'_v)
        """
        prior_llr = base_prior_llr.copy()
        vars_on_check = self.check_to_var[frustrated_check]

        if not vars_on_check:
            return prior_llr

        p = float(np.clip(p_error, self.epsilon, 1.0 - self.epsilon))

        for v in vars_on_check:
            delta = float(self.rng.uniform(0.0, self.perturb_strength))
            p_pert = ((1.0 + delta) * p) / (1.0 + delta * p)
            p_pert = float(np.clip(p_pert, self.epsilon, 1.0 - self.epsilon))
            prior_llr[v] = np.log((1.0 - p_pert) / p_pert)

        np.clip(prior_llr, -self.llr_clip, self.llr_clip, out=prior_llr)
        return prior_llr

    def _run_bp_phase(
        self,
        syndrome: np.ndarray,
        prior_llr: np.ndarray,
        max_phase_iters: int,
    ) -> tuple[np.ndarray, np.ndarray, bool, bool, int]:
        """
        Run one BP phase starting from fresh messages initialized by prior_llr.

        Returns:
            estimate,
            residual,
            success,
            converged,
            iterations_used_in_this_phase
        """
        sigma = np.where(syndrome == 1, -1.0, 1.0).astype(np.float64)

        mv_to_c = prior_llr[self.edge_var].copy()
        mc_to_v = np.zeros(self.num_edges, dtype=np.float64)

        bit_llr = prior_llr.copy()
        estimate = np.zeros(self.n, dtype=np.uint8)

        converged = False
        iters_used = 0

        for it in range(1, max_phase_iters + 1):
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
                np.clip(mc_to_v[start:end], -self.llr_clip, self.llr_clip, out=mc_to_v[start:end])

            bit_llr[:] = prior_llr
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

        return estimate.copy(), residual, success, converged, iters_used

    def decode(self, syndrome: np.ndarray, p_error: float) -> DecoderResult:
        """
        Flooding BP in LLR domain with optional binary random perturbation.

        Total iterations returned here count:
        - initial BP phase
        - plus all perturbation BP phases that were actually executed
        """
        syndrome = np.asarray(syndrome, dtype=np.uint8)
        if syndrome.shape != (self.m,):
            raise ValueError(f"syndrome must have shape ({self.m},)")

        base_prior_llr = self._base_prior_llr(p_error)

        estimate, residual, success, converged, phase_iters = self._run_bp_phase(
            syndrome=syndrome,
            prior_llr=base_prior_llr,
            max_phase_iters=self.max_iters,
        )

        total_iters = phase_iters

        if success or not self.enable_random_perturbation:
            return DecoderResult(
                estimated_error=estimate,
                success=success,
                iterations_used=total_iters,
                residual_syndrome=residual,
                converged=converged,
            )

        if self.perturb_iters <= 0 or self.perturb_max_feedbacks <= 0:
            return DecoderResult(
                estimated_error=estimate,
                success=success,
                iterations_used=total_iters,
                residual_syndrome=residual,
                converged=converged,
            )

        last_estimate = estimate
        last_residual = residual
        last_converged = converged
        last_success = success

        for _ in range(self.perturb_max_feedbacks):
            frustrated_checks = np.flatnonzero(last_residual)
            if frustrated_checks.size == 0:
                break

            chosen_check = int(self.rng.choice(frustrated_checks))
            perturbed_prior_llr = self._perturb_prior_llr(
                base_prior_llr=base_prior_llr,
                p_error=p_error,
                frustrated_check=chosen_check,
            )

            estimate, residual, success, converged, phase_iters = self._run_bp_phase(
                syndrome=syndrome,
                prior_llr=perturbed_prior_llr,
                max_phase_iters=self.perturb_iters,
            )

            total_iters += phase_iters

            last_estimate = estimate
            last_residual = residual
            last_converged = converged
            last_success = success

            if success:
                break

        return DecoderResult(
            estimated_error=last_estimate,
            success=last_success,
            iterations_used=total_iters,
            residual_syndrome=last_residual,
            converged=last_converged,
        )
"""Reward interfaces and concrete reward implementations for cell assignment RL."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .actions import Action, ActionType


class RewardFunction(Protocol):
    """Protocol for pluggable reward design."""

    def compute(
        self,
        previous_membership_mask: np.ndarray,
        action: Action,
        new_membership_mask: np.ndarray,
        done: bool,
        info: dict[str, Any],
    ) -> float:
        """Return scalar reward for one transition."""


class ZeroReward:
    """Default reward that keeps training logic decoupled from environment building."""

    def compute(
        self,
        previous_membership_mask: np.ndarray,
        action: Action,
        new_membership_mask: np.ndarray,
        done: bool,
        info: dict[str, Any],
    ) -> float:
        del previous_membership_mask, action, new_membership_mask, done, info
        return 0.0


def compute_reference_distribution(reference_counts: np.ndarray, epsilon: float) -> np.ndarray:
    """Compute theta[k, g] = (C[k, g] + eps) / sum_g(C[k, g] + eps)."""
    counts = np.asarray(reference_counts, dtype=np.float64)

    if counts.ndim != 2:
        raise ValueError("reference_counts must have shape (K, G)")

    if counts.shape[0] == 0 or counts.shape[1] == 0:
        raise ValueError("reference_counts must have positive shape in both dimensions")

    if epsilon < 0:
        raise ValueError("epsilon must be >= 0")

    adjusted = counts + float(epsilon)
    row_sums = adjusted.sum(axis=1, keepdims=True)

    if np.any(row_sums <= 0):
        raise ValueError("each cell type row must have positive total count after epsilon")

    theta = adjusted / row_sums
    return theta.astype(np.float64)


def compute_bin_log_likelihood_by_type(bin_counts: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Compute LL[b, k] = (1/Nb) * sum_g x[b,g] * log(theta[k,g]).

    By definition, if Nb == 0 then LL[b, k] = 0 for all k.
    """
    x = np.asarray(bin_counts, dtype=np.float64)
    theta_arr = np.asarray(theta, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError("bin_counts must have shape (B, G)")

    if theta_arr.ndim != 2:
        raise ValueError("theta must have shape (K, G)")

    if x.shape[1] != theta_arr.shape[1]:
        raise ValueError("gene dimension mismatch between bin_counts and theta")

    if np.any(x < 0):
        raise ValueError("bin_counts must be non-negative")

    if np.any(theta_arr <= 0) or np.any(~np.isfinite(theta_arr)):
        raise ValueError("theta must be finite and strictly positive")

    log_theta = np.log(theta_arr)  # (K, G)
    # Use einsum instead of BLAS-backed matmul to avoid OpenMP runtime issues
    # in restricted cluster/sandbox environments.
    weighted = np.einsum("bg,kg->bk", x, log_theta, optimize=False)  # (B, K)
    nb = x.sum(axis=1, keepdims=True)  # (B, 1)

    ll = np.zeros_like(weighted, dtype=np.float64)
    positive = nb[:, 0] > 0
    if np.any(positive):
        ll[positive] = weighted[positive] / nb[positive]

    # Keep LL=0 for bins with Nb==0 by definition.
    return ll


def _softmax_stable(scores: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over 1D scores."""
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError("scores must be a 1D array")

    max_s = np.max(s)
    exp_s = np.exp(s - max_s)
    denom = np.sum(exp_s)
    if denom <= 0 or not np.isfinite(denom):
        raise ValueError("softmax denominator became non-finite or non-positive")

    return exp_s / denom


class PosteriorAddBinReward:
    """Reward model based on cell-type posterior + distance + overlap penalties.

    Implements the exact formula family:
    - theta[k,g] from reference counts with pseudocount epsilon
    - LL[b,k] per bin/type normalized by bin total counts
    - p(k|ST) from softmax(log p(ST|k) + uniform prior)
    - R_add[b] = w1 * R_expr[b] - w2 * P_dis[b] - w3 * P_overlap[b]
    - R_stop = -lambda * max_b R_add[b] over currently add-eligible bins
    """

    def __init__(
        self,
        reference_counts: np.ndarray,
        candidate_bin_ids: list[str] | tuple[str, ...],
        candidate_expression: np.ndarray,
        candidate_bin_xy_um: np.ndarray,
        nucleus_center_xy_um: np.ndarray,
        other_nuclei_center_xy_um: np.ndarray | None = None,
        *,
        epsilon: float = 1e-8,
        r_max_um: float = 80.0,
        w1: float = 1.0,
        w2: float = 1.0,
        w3: float = 1.0,
        stop_lambda: float = 1.0,
        normalize_expression_zscore: bool = False,
        zscore_delta: float = 1e-8,
        remove_reward: float = 0.0,
    ) -> None:
        """Create reward object with all static per-episode inputs.

        Parameters
        ----------
        reference_counts:
            Array C[K, G] of reference gene counts by cell type.
        candidate_bin_ids:
            IDs for candidate bins in the same order as candidate_expression rows.
        candidate_expression:
            Array x[B, G] of per-bin gene counts.
        candidate_bin_xy_um:
            Array [B, 2] of bin centers in micrometres.
        nucleus_center_xy_um:
            Array [2] for current nucleus center.
        other_nuclei_center_xy_um:
            Optional array [N, 2] for other nuclei centers.
        """
        if r_max_um <= 0:
            raise ValueError("r_max_um must be > 0")

        if zscore_delta <= 0:
            raise ValueError("zscore_delta must be > 0")

        self._candidate_bin_ids = tuple(str(x) for x in candidate_bin_ids)
        self._bin_id_to_index = {bin_id: i for i, bin_id in enumerate(self._candidate_bin_ids)}
        if len(self._bin_id_to_index) != len(self._candidate_bin_ids):
            raise ValueError("candidate_bin_ids must be unique")

        self._x = np.asarray(candidate_expression, dtype=np.float64)
        self._bin_xy = np.asarray(candidate_bin_xy_um, dtype=np.float64)
        self._nucleus_center = np.asarray(nucleus_center_xy_um, dtype=np.float64)

        if self._x.ndim != 2:
            raise ValueError("candidate_expression must have shape (B, G)")

        if self._bin_xy.shape != (self._x.shape[0], 2):
            raise ValueError("candidate_bin_xy_um must have shape (B, 2)")

        if self._nucleus_center.shape != (2,):
            raise ValueError("nucleus_center_xy_um must have shape (2,)")

        if np.any(self._x < 0):
            raise ValueError("candidate_expression must be non-negative")

        self._theta = compute_reference_distribution(reference_counts=reference_counts, epsilon=epsilon)

        if self._theta.shape[1] != self._x.shape[1]:
            raise ValueError(
                "gene dimension mismatch: reference_counts has G=%d but candidate_expression has G=%d"
                % (self._theta.shape[1], self._x.shape[1])
            )

        self._ll = compute_bin_log_likelihood_by_type(bin_counts=self._x, theta=self._theta)  # (B, K)
        self._n_cell_types = int(self._theta.shape[0])
        self._log_prior = -np.log(float(self._n_cell_types))

        # Precompute distance terms for all candidate bins.
        # Use explicit sqrt(sum(square)) instead of np.linalg.norm to avoid
        # BLAS/OpenMP backend issues in some runtime environments.
        delta_n = self._bin_xy - self._nucleus_center[None, :]
        self._d_n = np.sqrt(np.sum(delta_n * delta_n, axis=1))

        if other_nuclei_center_xy_um is None:
            self._d_other = np.full(self._x.shape[0], np.inf, dtype=np.float64)
        else:
            other = np.asarray(other_nuclei_center_xy_um, dtype=np.float64)
            if other.ndim != 2 or (other.shape[1] != 2):
                raise ValueError("other_nuclei_center_xy_um must have shape (N, 2)")

            if other.shape[0] == 0:
                self._d_other = np.full(self._x.shape[0], np.inf, dtype=np.float64)
            else:
                # Compute nearest distance to any other nucleus per bin.
                deltas = self._bin_xy[:, None, :] - other[None, :, :]
                dists = np.sqrt(np.sum(deltas * deltas, axis=2))
                self._d_other = np.min(dists, axis=1)

        self._r_max_um = float(r_max_um)
        self._w1 = float(w1)
        self._w2 = float(w2)
        self._w3 = float(w3)
        self._stop_lambda = float(stop_lambda)
        self._normalize_expression_zscore = bool(normalize_expression_zscore)
        self._zscore_delta = float(zscore_delta)
        self._remove_reward = float(remove_reward)
        self._p_dis = self._d_n / self._r_max_um
        self._p_overlap = np.maximum(0.0, (self._d_n - self._d_other) / self._r_max_um)

    @property
    def n_cell_types(self) -> int:
        """Return K, number of reference cell types."""
        return self._n_cell_types

    @property
    def n_candidate_bins(self) -> int:
        """Return B, number of candidate bins in this episode."""
        return int(self._x.shape[0])

    def posterior_given_state(self, membership_mask: np.ndarray) -> np.ndarray:
        """Compute p(k|ST) using uniform prior and softmax in log space."""
        mask = self._validate_membership_mask(membership_mask)

        if np.any(mask == 1):
            log_p_st_given_k = self._ll[mask == 1].sum(axis=0)
        else:
            # Empty ST: log p(ST|k) = 0 for all k.
            log_p_st_given_k = np.zeros(self.n_cell_types, dtype=np.float64)

        s = log_p_st_given_k + self._log_prior
        return _softmax_stable(s)

    def expression_reward_per_bin(self, membership_mask: np.ndarray) -> np.ndarray:
        """Compute R_expr[b] = sum_k p(k|ST) * LL[b,k] for all bins."""
        posterior = self.posterior_given_state(membership_mask)
        return np.sum(self._ll * posterior[None, :], axis=1)

    def add_reward_for_bin(self, membership_mask: np.ndarray, bin_id: str) -> float:
        """Compute R_add for one bin only.

        This helper avoids full-vector recomputation for ADD_BIN when expression
        z-score normalization is disabled.
        """
        mask = self._validate_membership_mask(membership_mask)
        if bin_id not in self._bin_id_to_index:
            raise KeyError(f"unknown bin_id for reward: {bin_id!r}")

        idx = self._bin_id_to_index[bin_id]
        posterior = self.posterior_given_state(mask)
        return self._add_reward_for_index(mask, idx, posterior)

    def add_reward_per_bin(self, membership_mask: np.ndarray) -> np.ndarray:
        """Compute R_add[b] for all candidate bins given current state ST."""
        mask = self._validate_membership_mask(membership_mask)

        posterior = self.posterior_given_state(mask)
        r_expr = np.sum(self._ll * posterior[None, :], axis=1)

        if self._normalize_expression_zscore:
            eligible = mask == 0
            if np.any(eligible):
                mu = float(np.mean(r_expr[eligible]))
                sigma = float(np.std(r_expr[eligible], ddof=0))
                expr_term = (r_expr - mu) / (sigma + self._zscore_delta)
            else:
                expr_term = np.zeros_like(r_expr)
        else:
            expr_term = r_expr

        r_add = self._w1 * expr_term - self._w2 * self._p_dis - self._w3 * self._p_overlap
        return r_add

    def stop_reward(self, membership_mask: np.ndarray) -> float:
        """Compute R_stop = -lambda * max_b R_add[b] over currently add-eligible bins."""
        mask = self._validate_membership_mask(membership_mask)
        eligible = mask == 0

        if not np.any(eligible):
            delta_t = 0.0
        else:
            r_add = self.add_reward_per_bin(mask)
            delta_t = float(np.max(r_add[eligible]))

        return -self._stop_lambda * delta_t

    def evaluate_state(self, membership_mask: np.ndarray) -> dict[str, np.ndarray | float]:
        """Return p(k|ST), R_add per bin, and R_stop for one state ST."""
        mask = self._validate_membership_mask(membership_mask)
        p_k_given_st = self.posterior_given_state(mask)
        r_add = self.add_reward_per_bin(mask)
        r_stop = self.stop_reward(mask)

        return {
            "p_k_given_ST": p_k_given_st,
            "R_add": r_add,
            "R_stop": float(r_stop),
        }

    def compute(
        self,
        previous_membership_mask: np.ndarray,
        action: Action,
        new_membership_mask: np.ndarray,
        done: bool,
        info: dict[str, Any],
    ) -> float:
        """Return scalar reward for one transition.

        Reward semantics:
        - ADD_BIN: use R_add[b] computed from previous state ST.
        - STOP: use R_stop computed from previous state ST.
        - REMOVE_BIN: returns configurable constant `remove_reward`.
        """
        del new_membership_mask, done, info

        prev_mask = self._validate_membership_mask(previous_membership_mask)

        if action.kind == ActionType.ADD_BIN:
            if action.bin_id is None:
                raise ValueError("ADD_BIN action requires bin_id")
            if action.bin_id not in self._bin_id_to_index:
                raise KeyError(f"unknown bin_id for reward: {action.bin_id!r}")

            idx = self._bin_id_to_index[action.bin_id]
            posterior = self.posterior_given_state(prev_mask)
            return self._add_reward_for_index(prev_mask, idx, posterior)

        if action.kind == ActionType.STOP:
            return float(self.stop_reward(prev_mask))

        if action.kind == ActionType.REMOVE_BIN:
            return self._remove_reward

        raise ValueError(f"unsupported action kind for reward: {action.kind!r}")

    def _validate_membership_mask(self, membership_mask: np.ndarray) -> np.ndarray:
        """Validate membership mask shape and domain and return int8 view."""
        mask = np.asarray(membership_mask, dtype=np.int8)

        if mask.shape != (self.n_candidate_bins,):
            raise ValueError(
                "membership_mask must have shape (B,), got %s expected (%d,)"
                % (mask.shape, self.n_candidate_bins)
            )

        unique = np.unique(mask)
        if not set(unique.tolist()).issubset({0, 1}):
            raise ValueError("membership_mask must contain only 0/1 values")

        return mask

    def _add_reward_for_index(
        self,
        membership_mask: np.ndarray,
        bin_index: int,
        posterior: np.ndarray,
    ) -> float:
        """Internal single-bin reward computation.

        Fast path:
        - If z-score normalization is disabled, computes in O(K) time.
        Fallback:
        - If z-score normalization is enabled, computes required eligible-bin
          statistics first.
        """
        expr = float(np.sum(self._ll[bin_index] * posterior))

        if self._normalize_expression_zscore:
            r_expr = np.sum(self._ll * posterior[None, :], axis=1)
            eligible = membership_mask == 0
            if np.any(eligible):
                mu = float(np.mean(r_expr[eligible]))
                sigma = float(np.std(r_expr[eligible], ddof=0))
                expr_term = (expr - mu) / (sigma + self._zscore_delta)
            else:
                expr_term = 0.0
        else:
            expr_term = expr

        return float(
            self._w1 * expr_term
            - self._w2 * self._p_dis[bin_index]
            - self._w3 * self._p_overlap[bin_index]
        )

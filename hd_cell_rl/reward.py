"""Reward interfaces and concrete reward implementations for cell assignment RL."""

from __future__ import annotations

import re
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


_SQUARE_BARCODE_RE = re.compile(r"^s_\d+um_(\d+)_(\d+)(?:-\d+)?$")


def _estimate_grid_step(coords: np.ndarray) -> float:
    """Estimate one axis spacing for fallback neighbor construction from XY coordinates."""
    vals = np.unique(np.asarray(coords, dtype=np.float64))
    if vals.size <= 1:
        return 1.0
    diffs = np.diff(np.sort(vals))
    diffs = diffs[np.isfinite(diffs) & (diffs > 1.0e-6)]
    if diffs.size == 0:
        return 1.0
    return float(np.min(diffs))


def build_eight_neighbor_index(
    candidate_bin_ids: list[str] | tuple[str, ...],
    candidate_bin_xy_um: np.ndarray,
) -> np.ndarray:
    """Build fixed-width 8-neighbor index table for candidate bins.

    Returns
    -------
    np.ndarray
        Shape (B, 8) int32 array. Missing neighbors are encoded as -1.
    """
    bin_ids = tuple(str(x) for x in candidate_bin_ids)
    xy = np.asarray(candidate_bin_xy_um, dtype=np.float64)
    n_bins = len(bin_ids)
    if xy.shape != (n_bins, 2):
        raise ValueError("candidate_bin_xy_um must have shape (B, 2)")

    neighbor_index = np.full((n_bins, 8), -1, dtype=np.int32)
    if n_bins == 0:
        return neighbor_index

    grid_coords: list[tuple[int, int]] = []
    all_parsed = True
    for bin_id in bin_ids:
        m = _SQUARE_BARCODE_RE.match(bin_id)
        if m is None:
            all_parsed = False
            break
        grid_coords.append((int(m.group(1)), int(m.group(2))))

    if not all_parsed:
        step_x = _estimate_grid_step(xy[:, 0])
        step_y = _estimate_grid_step(xy[:, 1])
        x0 = float(np.min(xy[:, 0]))
        y0 = float(np.min(xy[:, 1]))
        gx = np.rint((xy[:, 0] - x0) / step_x).astype(np.int64)
        gy = np.rint((xy[:, 1] - y0) / step_y).astype(np.int64)
        grid_coords = list(zip(gx.tolist(), gy.tolist()))

    coord_to_index = {coord: idx for idx, coord in enumerate(grid_coords)}
    offsets = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    for idx, (gx, gy) in enumerate(grid_coords):
        for nbr_pos, (dx, dy) in enumerate(offsets):
            nbr_idx = coord_to_index.get((gx + dx, gy + dy))
            if nbr_idx is not None:
                neighbor_index[idx, nbr_pos] = int(nbr_idx)

    return neighbor_index


def compute_neighbor_support_fraction(
    membership_mask: np.ndarray,
    neighbor_index: np.ndarray,
) -> np.ndarray:
    """Return per-bin fraction of already-assigned 8-neighbors."""
    mask = np.asarray(membership_mask, dtype=np.uint8)
    neighbors = np.asarray(neighbor_index, dtype=np.int32)
    if mask.ndim != 1:
        raise ValueError("membership_mask must be a 1D array")
    if neighbors.shape != (mask.shape[0], 8):
        raise ValueError("neighbor_index must have shape (B, 8)")

    padded = np.zeros(mask.shape[0] + 1, dtype=np.uint8)
    padded[: mask.shape[0]] = mask
    safe_neighbors = np.where(neighbors >= 0, neighbors, mask.shape[0])
    touched = padded[safe_neighbors].sum(axis=1, dtype=np.uint8)
    return touched.astype(np.float64) / 8.0


def compute_expression_confidence(
    bin_count_totals: np.ndarray,
    pseudocount: float,
) -> np.ndarray:
    """Return per-bin confidence c / (c + a) from total selected-gene counts."""
    counts = np.asarray(bin_count_totals, dtype=np.float64)
    if counts.ndim != 1:
        raise ValueError("bin_count_totals must be a 1D array")
    if np.any(counts < 0):
        raise ValueError("bin_count_totals must be non-negative")
    if pseudocount < 0:
        raise ValueError("pseudocount must be >= 0")

    denom = counts + float(pseudocount)
    out = np.zeros_like(counts, dtype=np.float64)
    positive = denom > 0
    if np.any(positive):
        out[positive] = counts[positive] / denom[positive]
    return out


def compute_frontier_eligible_mask(
    membership_mask: np.ndarray,
    neighbor_index: np.ndarray,
) -> np.ndarray:
    """Return frontier-only add eligibility mask.

    A bin is frontier-eligible when:
    - it is currently unassigned, and
    - it touches at least one already-assigned 8-neighbor.

    Special case:
    - if no bins are assigned yet, all unassigned bins are eligible.
    """
    mask = np.asarray(membership_mask, dtype=np.uint8)
    if mask.ndim != 1:
        raise ValueError("membership_mask must be a 1D array")
    unassigned = mask == 0
    if mask.size == 0:
        return unassigned
    if not np.any(mask == 1):
        return unassigned
    support = compute_neighbor_support_fraction(mask, neighbor_index)
    return unassigned & (support > 0.0)


def compute_stop_delta(
    add_rewards: np.ndarray,
    eligible_mask: np.ndarray,
    *,
    stop_stat: str = "max",
    stop_top_k: int = 3,
) -> float:
    """Summarize remaining add opportunities for STOP reward.

    Parameters
    ----------
    add_rewards:
        Per-bin ADD rewards.
    eligible_mask:
        Boolean mask over add-eligible bins.
    stop_stat:
        One of:
        - ``max``: use the single best eligible ADD reward.
        - ``topk_mean``: use the mean of the top-k eligible ADD rewards.
    stop_top_k:
        Number of best eligible rewards to average when ``stop_stat='topk_mean'``.
        If fewer than k bins are eligible, average all eligible bins.
    """
    rewards = np.asarray(add_rewards, dtype=np.float64)
    eligible = np.asarray(eligible_mask, dtype=bool)
    if rewards.ndim != 1:
        raise ValueError("add_rewards must be a 1D array")
    if eligible.shape != rewards.shape:
        raise ValueError("eligible_mask must have the same shape as add_rewards")
    if not np.any(eligible):
        return 0.0

    eligible_rewards = rewards[eligible]
    if stop_stat == "max":
        return float(np.max(eligible_rewards))
    if stop_stat == "topk_mean":
        if stop_top_k <= 0:
            raise ValueError("stop_top_k must be > 0 when stop_stat='topk_mean'")
        k = int(min(stop_top_k, eligible_rewards.shape[0]))
        if k == eligible_rewards.shape[0]:
            return float(np.mean(eligible_rewards))
        top_idx = np.argpartition(eligible_rewards, -k)[-k:]
        return float(np.mean(eligible_rewards[top_idx]))
    raise ValueError(f"unsupported stop_stat: {stop_stat!r}")


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
    - conf[b] = c[b] / (c[b] + a), where c[b] is selected-gene total count in bin b
    - p(k|ST) from softmax(log p(ST|k) + uniform prior)
    - neighbor_support[b] = touched_8_neighbors[b] / 8
    - R_add[b] = w1 * (conf[b] * R_expr[b]) - w2 * P_dis[b] - w3 * P_overlap[b] + w4 * neighbor_support[b]
    - R_stop = -lambda * stop_delta over currently add-eligible bins
      where stop_delta is either max frontier add reward or mean(top-k frontier add rewards)
    """

    def __init__(
        self,
        reference_counts: np.ndarray,
        candidate_bin_ids: list[str] | tuple[str, ...],
        candidate_expression: np.ndarray | None,
        candidate_bin_xy_um: np.ndarray,
        nucleus_center_xy_um: np.ndarray,
        other_nuclei_center_xy_um: np.ndarray | None = None,
        *,
        epsilon: float = 1e-8,
        r_max_um: float = 80.0,
        w1: float = 1.0,
        w2: float = 1.0,
        w3: float = 1.0,
        w4: float = 0.0,
        stop_lambda: float = 1.0,
        stop_stat: str = "max",
        stop_top_k: int = 3,
        expression_confidence_pseudocount: float = 5.0,
        normalize_expression_zscore: bool = False,
        zscore_delta: float = 1e-8,
        remove_reward: float = 0.0,
        precomputed_ll: np.ndarray | None = None,
        precomputed_bin_count_totals: np.ndarray | None = None,
        precomputed_d_other_um: np.ndarray | None = None,
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
        if expression_confidence_pseudocount < 0:
            raise ValueError("expression_confidence_pseudocount must be >= 0")

        self._candidate_bin_ids = tuple(str(x) for x in candidate_bin_ids)
        self._bin_id_to_index = {bin_id: i for i, bin_id in enumerate(self._candidate_bin_ids)}
        if len(self._bin_id_to_index) != len(self._candidate_bin_ids):
            raise ValueError("candidate_bin_ids must be unique")

        self._bin_xy = np.asarray(candidate_bin_xy_um, dtype=np.float64)
        self._nucleus_center = np.asarray(nucleus_center_xy_um, dtype=np.float64)
        n_bins = len(self._candidate_bin_ids)

        if self._bin_xy.shape != (n_bins, 2):
            raise ValueError("candidate_bin_xy_um must have shape (B, 2)")

        if self._nucleus_center.shape != (2,):
            raise ValueError("nucleus_center_xy_um must have shape (2,)")

        bin_count_totals: np.ndarray
        if precomputed_ll is not None:
            ll = np.asarray(precomputed_ll, dtype=np.float64)
            if ll.ndim != 2:
                raise ValueError("precomputed_ll must have shape (B, K)")
            if ll.shape[0] != n_bins:
                raise ValueError(
                    "precomputed_ll row count must match number of candidate bins: %d != %d"
                    % (ll.shape[0], n_bins)
                )
            if not np.isfinite(ll).all():
                raise ValueError("precomputed_ll contains non-finite values")
            self._ll = ll
            self._n_cell_types = int(ll.shape[1])
            if self._n_cell_types <= 0:
                raise ValueError("precomputed_ll must have positive K dimension")
            # Keep reference compatibility check on K even in precomputed mode.
            ref_counts = np.asarray(reference_counts, dtype=np.float64)
            if ref_counts.ndim != 2 or ref_counts.shape[0] != self._n_cell_types:
                raise ValueError(
                    "reference_counts K dimension must match precomputed_ll: %d != %d"
                    % (ref_counts.shape[0] if ref_counts.ndim == 2 else -1, self._n_cell_types)
                )
            if precomputed_bin_count_totals is None:
                raise ValueError("precomputed_bin_count_totals must be provided when precomputed_ll is set")
            bin_count_totals = np.asarray(precomputed_bin_count_totals, dtype=np.float64)
        else:
            if candidate_expression is None:
                raise ValueError("candidate_expression must be provided when precomputed_ll is not set")
            x = np.asarray(candidate_expression, dtype=np.float64)
            if x.ndim != 2:
                raise ValueError("candidate_expression must have shape (B, G)")
            if x.shape[0] != n_bins:
                raise ValueError("candidate_expression row count must match number of candidate bins")
            if np.any(x < 0):
                raise ValueError("candidate_expression must be non-negative")

            theta = compute_reference_distribution(reference_counts=reference_counts, epsilon=epsilon)
            if theta.shape[1] != x.shape[1]:
                raise ValueError(
                    "gene dimension mismatch: reference_counts has G=%d but candidate_expression has G=%d"
                    % (theta.shape[1], x.shape[1])
                )
            self._ll = compute_bin_log_likelihood_by_type(bin_counts=x, theta=theta)  # (B, K)
            self._n_cell_types = int(theta.shape[0])
            bin_count_totals = np.sum(x, axis=1, dtype=np.float64)

        if bin_count_totals.shape != (n_bins,):
            raise ValueError(
                "bin count totals must have shape (B,), got %r for B=%d" % (bin_count_totals.shape, n_bins)
            )
        if np.any(bin_count_totals < 0):
            raise ValueError("bin count totals must be non-negative")

        self._log_prior = -np.log(float(self._n_cell_types))

        # Precompute distance terms for all candidate bins.
        # Use explicit sqrt(sum(square)) instead of np.linalg.norm to avoid
        # BLAS/OpenMP backend issues in some runtime environments.
        delta_n = self._bin_xy - self._nucleus_center[None, :]
        self._d_n = np.sqrt(np.sum(delta_n * delta_n, axis=1))

        if precomputed_d_other_um is not None:
            d_other = np.asarray(precomputed_d_other_um, dtype=np.float64)
            if d_other.shape != (n_bins,):
                raise ValueError(
                    "precomputed_d_other_um must have shape (B,), got %r for B=%d"
                    % (d_other.shape, n_bins)
                )
            if not np.isfinite(d_other).all():
                raise ValueError("precomputed_d_other_um contains non-finite values")
            if (d_other < 0).any():
                raise ValueError("precomputed_d_other_um must be non-negative")
            self._d_other = d_other
        elif other_nuclei_center_xy_um is None:
            self._d_other = np.full(n_bins, np.inf, dtype=np.float64)
        else:
            other = np.asarray(other_nuclei_center_xy_um, dtype=np.float64)
            if other.ndim != 2 or (other.shape[1] != 2):
                raise ValueError("other_nuclei_center_xy_um must have shape (N, 2)")

            if other.shape[0] == 0:
                self._d_other = np.full(n_bins, np.inf, dtype=np.float64)
            else:
                # Compute nearest distance to any other nucleus per bin.
                deltas = self._bin_xy[:, None, :] - other[None, :, :]
                dists = np.sqrt(np.sum(deltas * deltas, axis=2))
                self._d_other = np.min(dists, axis=1)

        self._r_max_um = float(r_max_um)
        self._w1 = float(w1)
        self._w2 = float(w2)
        self._w3 = float(w3)
        self._w4 = float(w4)
        self._stop_lambda = float(stop_lambda)
        self._stop_stat = str(stop_stat).strip().lower()
        if self._stop_stat not in {"max", "topk_mean"}:
            raise ValueError("stop_stat must be 'max' or 'topk_mean'")
        self._stop_top_k = int(stop_top_k)
        if self._stop_top_k <= 0:
            raise ValueError("stop_top_k must be > 0")
        self._normalize_expression_zscore = bool(normalize_expression_zscore)
        self._zscore_delta = float(zscore_delta)
        self._remove_reward = float(remove_reward)
        self._expression_confidence = compute_expression_confidence(
            bin_count_totals=bin_count_totals,
            pseudocount=float(expression_confidence_pseudocount),
        )
        self._p_dis = self._d_n / self._r_max_um
        self._p_overlap = np.maximum(0.0, (self._d_n - self._d_other) / self._r_max_um)
        self._neighbor_index = build_eight_neighbor_index(self._candidate_bin_ids, self._bin_xy)

    @property
    def n_cell_types(self) -> int:
        """Return K, number of reference cell types."""
        return self._n_cell_types

    @property
    def n_candidate_bins(self) -> int:
        """Return B, number of candidate bins in this episode."""
        return int(self._ll.shape[0])

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
        return np.sum(self._ll * posterior[None, :], axis=1) * self._expression_confidence

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
        eligible = self.frontier_add_mask(mask)

        posterior = self.posterior_given_state(mask)
        r_expr = np.sum(self._ll * posterior[None, :], axis=1) * self._expression_confidence

        if self._normalize_expression_zscore:
            if np.any(eligible):
                mu = float(np.mean(r_expr[eligible]))
                sigma = float(np.std(r_expr[eligible], ddof=0))
                expr_term = (r_expr - mu) / (sigma + self._zscore_delta)
            else:
                expr_term = np.zeros_like(r_expr)
        else:
            expr_term = r_expr

        neighbor_support = compute_neighbor_support_fraction(mask, self._neighbor_index)
        r_add = (
            self._w1 * expr_term
            - self._w2 * self._p_dis
            - self._w3 * self._p_overlap
            + self._w4 * neighbor_support
        )
        return r_add

    def stop_reward(self, membership_mask: np.ndarray) -> float:
        """Compute R_stop from remaining eligible ADD rewards."""
        delta_t = self.stop_delta(membership_mask)
        return -self._stop_lambda * delta_t

    def stop_delta(self, membership_mask: np.ndarray) -> float:
        """Return summarized remaining ADD opportunity used by STOP logic."""
        mask = self._validate_membership_mask(membership_mask)
        eligible = self.frontier_add_mask(mask)

        r_add = self.add_reward_per_bin(mask)
        return compute_stop_delta(
            r_add,
            eligible,
            stop_stat=self._stop_stat,
            stop_top_k=self._stop_top_k,
        )

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
            "eligible_add_mask": self.frontier_add_mask(mask),
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

    def frontier_add_mask(self, membership_mask: np.ndarray) -> np.ndarray:
        """Return frontier-only add eligibility mask for the current state."""
        mask = self._validate_membership_mask(membership_mask)
        return compute_frontier_eligible_mask(mask, self._neighbor_index)

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
        expr = float(np.sum(self._ll[bin_index] * posterior) * self._expression_confidence[bin_index])

        if self._normalize_expression_zscore:
            r_expr = np.sum(self._ll * posterior[None, :], axis=1) * self._expression_confidence
            eligible = self.frontier_add_mask(membership_mask)
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
            + self._w4 * self._neighbor_fraction_for_index(membership_mask, bin_index)
        )

    def _neighbor_fraction_for_index(self, membership_mask: np.ndarray, bin_index: int) -> float:
        """Return touched-8-neighbor fraction for one candidate bin."""
        neighbors = self._neighbor_index[bin_index]
        valid = neighbors >= 0
        if not np.any(valid):
            return 0.0
        return float(np.sum(membership_mask[neighbors[valid]], dtype=np.float64) / 8.0)

"""Actor-critic model used by PPO/GRPO training and evaluation."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from .ppo_feature_schema import (
    A_FEATURE_1,
    A_FEATURE_2,
    A_FEATURE_3,
    A_FEATURE_4,
    A_FEATURE_5,
    A_FEATURE_6,
    A_FEATURE_7,
    A_FEATURE_8,
    A_FEATURE_9,
    A_FEATURE_10,
    A_IS_STOP_ACTION,
    G_ASSIGNED_FRAC,
    G_ASSIGNED_LL_MEAN,
    G_CENTROID_DRIFT_SCALED,
    G_COMPACTNESS_PROXY,
    G_GROW_RATIO_SCALED,
    G_N_BINS_SCALED,
    G_POSITIVE_FRONTIER_FRACTION,
    G_REMAINING_FRAC,
    G_SEED_SIZE_SCALED,
    G_STEP_FRAC,
)

DEFAULT_PLANNER_MODE_COUNT = 4


class ActorCritic(nn.Module):
    """Masked actor-critic model over variable-size action sets."""

    def __init__(
        self,
        global_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        *,
        planner_enabled: bool = False,
        planner_mode_count: int = DEFAULT_PLANNER_MODE_COUNT,
    ) -> None:
        super().__init__()
        self.global_dim = int(global_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.planner_enabled = bool(planner_enabled)
        self.planner_mode_count = int(planner_mode_count)
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.logit_head = nn.Linear(hidden_dim * 2, 1)
        self.value_head = nn.Linear(hidden_dim, 1)
        if self.planner_enabled:
            self.planner_head = nn.Linear(hidden_dim, self.planner_mode_count)

    def encode_global(self, global_features: torch.Tensor) -> torch.Tensor:
        """Encode state-level features for critic/value prediction."""
        return self.global_encoder(global_features)

    def encode_action_features(self, action_features: torch.Tensor) -> torch.Tensor:
        """Encode per-action features for masked actor scoring."""
        return self.action_encoder(action_features)

    def value_from_global_latent(self, global_latent: torch.Tensor) -> torch.Tensor:
        """Predict scalar state value from encoded global features."""
        return self.value_head(global_latent).squeeze(-1)

    def policy_logits_from_action_latent(self, action_latent: torch.Tensor) -> torch.Tensor:
        """Score actions from action latents only.

        The original actor head is linear over [action_latent, global_latent]. The
        global half contributes the same scalar offset to every action within one
        state, so it cancels exactly inside the categorical softmax. Using only the
        action half is therefore mathematically equivalent for policy probabilities
        while avoiding repeated global-action concatenation.
        """
        actor_weight = self.logit_head.weight[:, : self.hidden_dim]
        return F.linear(action_latent, actor_weight, self.logit_head.bias).squeeze(-1)

    def stop_logits_from_global_features(self, global_features: torch.Tensor) -> torch.Tensor:
        """Build the STOP row from global features and score it with the actor."""
        if global_features.ndim != 2 or global_features.shape[1] != self.global_dim:
            raise ValueError(
                f"global_features must have shape (N, {self.global_dim}) to score STOP logits"
            )
        stop_features = global_features.new_zeros((global_features.shape[0], self.action_dim))
        stop_features[:, A_IS_STOP_ACTION] = 1.0
        stop_features[:, A_FEATURE_1] = global_features[:, G_ASSIGNED_FRAC]
        stop_features[:, A_FEATURE_2] = global_features[:, G_STEP_FRAC]
        stop_features[:, A_FEATURE_3] = global_features[:, G_N_BINS_SCALED]
        stop_features[:, A_FEATURE_4] = global_features[:, G_ASSIGNED_LL_MEAN]
        stop_features[:, A_FEATURE_5] = global_features[:, G_REMAINING_FRAC]
        stop_features[:, A_FEATURE_6] = global_features[:, G_SEED_SIZE_SCALED]
        stop_features[:, A_FEATURE_7] = global_features[:, G_GROW_RATIO_SCALED]
        stop_features[:, A_FEATURE_8] = global_features[:, G_POSITIVE_FRONTIER_FRACTION]
        stop_features[:, A_FEATURE_9] = global_features[:, G_CENTROID_DRIFT_SCALED]
        stop_features[:, A_FEATURE_10] = global_features[:, G_COMPACTNESS_PROXY]
        stop_latent = self.encode_action_features(stop_features)
        return self.policy_logits_from_action_latent(stop_latent)

    def planner_distribution(self, global_features: torch.Tensor) -> Categorical:
        """Return high-level planner mode distribution for one or more states."""
        if not self.planner_enabled:
            raise RuntimeError("planner_distribution requires planner_enabled=True")
        if global_features.ndim != 2 or global_features.shape[1] != self.global_dim:
            raise ValueError(
                f"global_features must have shape (N, {self.global_dim}) to score planner modes"
            )
        g = self.encode_global(global_features)
        return Categorical(logits=self.planner_head(g))

    def forward(
        self,
        global_features: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
        action_logit_bias: torch.Tensor | None = None,
    ) -> tuple[Categorical, torch.Tensor]:
        """Return masked action distribution and state values.

        Parameters
        ----------
        global_features:
            Shape (N, G)
        action_features:
            Shape (N, A, F)
        action_mask:
            Shape (N, A), True for valid actions.
        """
        if global_features.ndim != 2:
            raise ValueError("global_features must have shape (N, G)")
        if action_features.ndim != 3:
            raise ValueError("action_features must have shape (N, A, F)")
        if action_mask.ndim != 2:
            raise ValueError("action_mask must have shape (N, A)")
        if action_features.shape[0] != global_features.shape[0]:
            raise ValueError("batch dimension mismatch between global_features and action_features")
        if action_mask.shape[0] != global_features.shape[0]:
            raise ValueError("batch dimension mismatch between global_features and action_mask")
        if action_mask.shape[1] != action_features.shape[1]:
            raise ValueError("action dimension mismatch between action_features and action_mask")

        n_batch, n_actions, _ = action_features.shape

        g = self.encode_global(global_features)  # (N, H)
        a = self.encode_action_features(action_features.reshape(n_batch * n_actions, -1)).reshape(n_batch, n_actions, -1)
        logits = self.policy_logits_from_action_latent(a)  # (N, A)
        if action_logit_bias is not None:
            if action_logit_bias.shape != logits.shape:
                raise ValueError(
                    f"action_logit_bias must have shape {tuple(logits.shape)}, got {tuple(action_logit_bias.shape)}"
                )
            logits = logits + action_logit_bias

        # Use a large negative value so invalid actions have effectively zero probability.
        neg_large = torch.finfo(logits.dtype).min
        masked_logits = logits.masked_fill(~action_mask, neg_large)
        dist = Categorical(logits=masked_logits)
        values = self.value_from_global_latent(g)  # (N,)
        return dist, values

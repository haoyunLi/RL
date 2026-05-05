"""Feature names and column indexes for PPO policy inputs."""

from __future__ import annotations

GLOBAL_FEATURE_NAMES: tuple[str, ...] = (
    "assigned_frac",
    "step_frac",
    "n_bins_scaled",
    "assigned_ll_mean",
    "assigned_ll_max",
    "remaining_frac",
    "seed_size_scaled",
    "grow_ratio_scaled",
    "positive_frontier_fraction",
    "centroid_drift_scaled",
    "compactness_proxy",
    "frontier_add_reward_topk_mean",
    "frontier_add_reward_max",
    "frontier_add_reward_mean",
    "compact_streak_scaled",
)
GLOBAL_FEATURE_INDEX: dict[str, int] = {name: i for i, name in enumerate(GLOBAL_FEATURE_NAMES)}
GLOBAL_FEATURE_DIM = len(GLOBAL_FEATURE_NAMES)

G_ASSIGNED_FRAC = GLOBAL_FEATURE_INDEX["assigned_frac"]
G_STEP_FRAC = GLOBAL_FEATURE_INDEX["step_frac"]
G_N_BINS_SCALED = GLOBAL_FEATURE_INDEX["n_bins_scaled"]
G_ASSIGNED_LL_MEAN = GLOBAL_FEATURE_INDEX["assigned_ll_mean"]
G_ASSIGNED_LL_MAX = GLOBAL_FEATURE_INDEX["assigned_ll_max"]
G_REMAINING_FRAC = GLOBAL_FEATURE_INDEX["remaining_frac"]
G_SEED_SIZE_SCALED = GLOBAL_FEATURE_INDEX["seed_size_scaled"]
G_GROW_RATIO_SCALED = GLOBAL_FEATURE_INDEX["grow_ratio_scaled"]
G_POSITIVE_FRONTIER_FRACTION = GLOBAL_FEATURE_INDEX["positive_frontier_fraction"]
G_CENTROID_DRIFT_SCALED = GLOBAL_FEATURE_INDEX["centroid_drift_scaled"]
G_COMPACTNESS_PROXY = GLOBAL_FEATURE_INDEX["compactness_proxy"]
G_FRONTIER_ADD_REWARD_TOPK_MEAN = GLOBAL_FEATURE_INDEX["frontier_add_reward_topk_mean"]
G_FRONTIER_ADD_REWARD_MAX = GLOBAL_FEATURE_INDEX["frontier_add_reward_max"]
G_FRONTIER_ADD_REWARD_MEAN = GLOBAL_FEATURE_INDEX["frontier_add_reward_mean"]
G_COMPACT_STREAK_SCALED = GLOBAL_FEATURE_INDEX["compact_streak_scaled"]

ACTION_FEATURE_NAMES: tuple[str, ...] = (
    "is_stop_action",
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4",
    "feature_5",
    "feature_6",
    "feature_7",
    "feature_8",
    "feature_9",
    "feature_10",
    "candidate_to_current_centroid_distance",
    "candidate_compactness_gain",
    "candidate_neighbor_support",
)
ACTION_FEATURE_INDEX: dict[str, int] = {name: i for i, name in enumerate(ACTION_FEATURE_NAMES)}
ACTION_FEATURE_DIM = len(ACTION_FEATURE_NAMES)

A_IS_STOP_ACTION = ACTION_FEATURE_INDEX["is_stop_action"]
A_FEATURE_1 = ACTION_FEATURE_INDEX["feature_1"]
A_FEATURE_2 = ACTION_FEATURE_INDEX["feature_2"]
A_FEATURE_3 = ACTION_FEATURE_INDEX["feature_3"]
A_FEATURE_4 = ACTION_FEATURE_INDEX["feature_4"]
A_FEATURE_5 = ACTION_FEATURE_INDEX["feature_5"]
A_FEATURE_6 = ACTION_FEATURE_INDEX["feature_6"]
A_FEATURE_7 = ACTION_FEATURE_INDEX["feature_7"]
A_FEATURE_8 = ACTION_FEATURE_INDEX["feature_8"]
A_FEATURE_9 = ACTION_FEATURE_INDEX["feature_9"]
A_FEATURE_10 = ACTION_FEATURE_INDEX["feature_10"]
A_CANDIDATE_CENTROID_DISTANCE = ACTION_FEATURE_INDEX["candidate_to_current_centroid_distance"]
A_CANDIDATE_COMPACTNESS_GAIN = ACTION_FEATURE_INDEX["candidate_compactness_gain"]
A_CANDIDATE_NEIGHBOR_SUPPORT = ACTION_FEATURE_INDEX["candidate_neighbor_support"]

STOP_ACTION_FEATURE_LABELS: tuple[str, ...] = (
    "is_stop_action",
    "assigned_frac",
    "step_frac",
    "n_bins_scaled",
    "assigned_ll_mean",
    "remaining_frac",
    "seed_size_scaled",
    "grow_ratio_scaled",
    "positive_frontier_fraction",
    "centroid_drift_scaled",
    "compactness_proxy",
    "candidate_to_current_centroid_distance_or_zero",
    "candidate_compactness_gain_or_zero",
    "candidate_neighbor_support_or_zero",
)

ADD_ACTION_FEATURE_LABELS: tuple[str, ...] = (
    "is_stop_action",
    "ll_mean_z",
    "ll_max_z",
    "p_dis",
    "p_overlap",
    "is_assigned",
    "seed_size_scaled_or_zero",
    "grow_ratio_scaled_or_zero",
    "positive_frontier_fraction_or_zero",
    "centroid_drift_scaled_or_zero",
    "compactness_proxy_or_zero",
    "candidate_to_current_centroid_distance",
    "candidate_compactness_gain",
    "candidate_neighbor_support",
)

"""Public API for the HD spatial-cell RL scaffold."""

from .actions import Action, ActionType
from .builder import build_episode_for_cell, build_episodes
from .config import EnvironmentConfig
from .episode_build import EpisodeBuildConfig, run_episode_build, run_episode_build_from_config
from .environment import CellAssignmentEnv
from .models import BinRecord, CellEpisodeData, NucleusRecord
from .policy import Policy, RandomPolicy
from .reward import (
    PosteriorAddBinReward,
    RewardFunction,
    ZeroReward,
    compute_bin_log_likelihood_by_type,
    compute_reference_distribution,
)
from .reward_grid_search import (
    GridAxis,
    RewardGridSearchConfig,
    run_reward_grid_search,
    run_reward_grid_search_from_config,
)
from .state import CellAssignmentState

try:
    from .ppo_training import (
        ActorCritic,
        PPOTrainingConfig,
        run_ppo_training,
        run_ppo_training_from_config,
    )
    _HAS_PPO = True
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    _HAS_PPO = False

__all__ = [
    "Action",
    "ActionType",
    "BinRecord",
    "CellAssignmentEnv",
    "CellAssignmentState",
    "CellEpisodeData",
    "EnvironmentConfig",
    "EpisodeBuildConfig",
    "NucleusRecord",
    "Policy",
    "PosteriorAddBinReward",
    "GridAxis",
    "RandomPolicy",
    "RewardFunction",
    "RewardGridSearchConfig",
    "ZeroReward",
    "build_episode_for_cell",
    "build_episodes",
    "run_episode_build",
    "run_episode_build_from_config",
    "run_reward_grid_search",
    "run_reward_grid_search_from_config",
    "compute_bin_log_likelihood_by_type",
    "compute_reference_distribution",
]

if _HAS_PPO:
    __all__.extend(
        [
            "PPOTrainingConfig",
            "ActorCritic",
            "run_ppo_training",
            "run_ppo_training_from_config",
        ]
    )

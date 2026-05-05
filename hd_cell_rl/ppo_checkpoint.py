"""Actor-critic construction and checkpoint loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .ppo_config import PPOTrainingConfig
from .ppo_feature_schema import ACTION_FEATURE_DIM, GLOBAL_FEATURE_DIM
from .ppo_model import ActorCritic


def load_checkpoint_payload(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load a checkpoint payload and validate its top-level shape."""
    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"invalid checkpoint payload in {path}")
    return payload


def build_actor_critic_from_config(
    config: PPOTrainingConfig,
    *,
    device: torch.device | None = None,
) -> ActorCritic:
    """Build the policy model shape implied by a resolved PPO config."""
    model = ActorCritic(
        global_dim=GLOBAL_FEATURE_DIM,
        action_dim=ACTION_FEATURE_DIM,
        hidden_dim=int(config.hidden_dim),
        planner_enabled=bool(config.planner_enabled),
        planner_mode_count=len(config.planner_modes),
    )
    if device is not None:
        model = model.to(device)
    return model


def load_actor_critic_checkpoint(
    checkpoint_path: str | Path,
    config: PPOTrainingConfig,
    *,
    device: torch.device,
    payload: dict[str, Any] | None = None,
) -> tuple[ActorCritic, dict[str, Any]]:
    """Load an actor-critic checkpoint using the model shape from ``config``."""
    path = Path(checkpoint_path).expanduser().resolve()
    if payload is None:
        payload = load_checkpoint_payload(path)
    if "model_state_dict" not in payload:
        raise ValueError(f"checkpoint missing 'model_state_dict': {path}")

    model = build_actor_critic_from_config(config, device=device)
    try:
        model.load_state_dict(payload["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"checkpoint model shape does not match config/state feature schema: {path}"
        ) from exc
    model.eval()
    return model, payload

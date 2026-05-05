"""Rollout collection for PPO/GRPO."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .ppo_config import (
    PLANNER_MODE_STOP,
    PLANNER_MODE_COMPACT,
    PLANNER_MODE_BALANCED,
    PPOTrainingConfig,
)
from .ppo_buffers import (
    EpisodeStep,
    PlannerStep,
    EpisodeTrajectory,
    RolloutTransition,
    PlannerRolloutTransition,
    RolloutFeatureCache,
    PlannerRolloutFeatureCache,
    compute_discounted_returns,
    compute_gae_returns_and_advantages,
    compute_advantages,
    _batch_global_features_for_rollout,
    _batch_observations_for_rollout,
    _build_planner_rollout_buffer,
    _build_planner_rollout_feature_cache,
    _build_rollout_buffer,
    _build_rollout_feature_cache,
    _compute_group_relative_episode_advantages,
    _pack_mask,
    _unpack_mask,
)
from .ppo_dataset import EpisodeDataset
from .ppo_model import ActorCritic
from .ppo_state import (
    EpisodeContext,
    _compute_state_feature_bundle,
    _observation_to_tensors,
    _observation_with_compact_streak,
    _observation_without_stop_action,
    _planner_logit_bias_from_action_features,
)

EnvFactory = Callable[[EpisodeContext], Any]


def run_episode(
    env: Any,
    model: ActorCritic,
    device: torch.device,
    rng: np.random.Generator,
    config: PPOTrainingConfig | None = None,
) -> EpisodeTrajectory:
    """Roll out exactly one episode and collect PPO-required fields."""
    if config is not None and bool(config.planner_enabled):
        return _run_episode_with_planner(env=env, model=model, device=device, rng=rng, config=config)

    obs, _ = env.reset()
    steps: list[EpisodeStep] = []
    total_reward = 0.0

    while True:
        global_t, action_t, mask_t = _observation_to_tensors(obs, device=device)
        with torch.inference_mode():
            dist, value = model(global_t, action_t, mask_t)
            probs = dist.probs.squeeze(0).detach().cpu().numpy()
            prob_sum = float(np.sum(probs))
            if prob_sum <= 0.0 or not np.isfinite(prob_sum):
                raise RuntimeError("action probabilities became non-finite or non-positive")
            probs = probs / prob_sum
            action = int(rng.choice(probs.shape[0], p=probs))
            action_tensor = torch.as_tensor([action], device=device, dtype=torch.int64)
            old_log_prob = float(dist.log_prob(action_tensor).item())

        old_value = float(value.item())

        packed_mask = _pack_mask(np.asarray(obs["membership_mask"], dtype=np.uint8))
        step_index = int(obs["step_index"])

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        total_reward += float(reward)
        steps.append(
            EpisodeStep(
                packed_membership_mask=packed_mask,
                step_index=step_index,
                action=action,
                reward=float(reward),
                done=done,
                old_log_prob=old_log_prob,
                old_value=old_value,
            )
        )
        obs = next_obs
        if done:
            break

    # Caller fills the actual episode slot index after collection.
    return EpisodeTrajectory(episode_slot=-1, steps=tuple(steps), total_reward=float(total_reward))


def _run_episode_with_planner(
    *,
    env: Any,
    model: ActorCritic,
    device: torch.device,
    rng: np.random.Generator,
    config: PPOTrainingConfig,
) -> EpisodeTrajectory:
    """Roll out one episode with high-level planner checkpoints controlling STOP/mode."""
    obs, _ = env.reset()
    steps: list[EpisodeStep] = []
    planner_steps: list[PlannerStep] = []
    total_reward = 0.0
    current_mode = PLANNER_MODE_BALANCED
    current_compact_streak = 0
    steps_since_planner = int(config.planner_interval)

    while True:
        needs_planner = not planner_steps or steps_since_planner >= int(config.planner_interval)
        if needs_planner:
            planner_obs = _observation_with_compact_streak(obs, current_compact_streak)
            global_t = torch.as_tensor(
                np.asarray(planner_obs["global_features"], dtype=np.float32),
                device=device,
            ).unsqueeze(0)
            with torch.inference_mode():
                planner_dist = model.planner_distribution(global_t)
                mode_t = planner_dist.sample()
                mode = int(mode_t.item())
                planner_log_prob = float(planner_dist.log_prob(mode_t).item())
            current_mode = mode
            steps_since_planner = 0
            planner_steps.append(
                PlannerStep(
                    packed_membership_mask=_pack_mask(np.asarray(obs["membership_mask"], dtype=np.uint8)),
                    step_index=int(obs["step_index"]),
                    mode=int(current_mode),
                    old_log_prob=planner_log_prob,
                    compact_streak=int(current_compact_streak),
                )
            )
            current_compact_streak = current_compact_streak + 1 if current_mode == PLANNER_MODE_COMPACT else 0

        if current_mode == PLANNER_MODE_STOP or not np.any(np.asarray(obs["action_mask"], dtype=bool)[1:]):
            packed_mask = _pack_mask(np.asarray(obs["membership_mask"], dtype=np.uint8))
            step_index = int(obs["step_index"])
            next_obs, reward, terminated, truncated, _ = env.step(0)
            done = bool(terminated or truncated)
            total_reward += float(reward)
            steps.append(
                EpisodeStep(
                    packed_membership_mask=packed_mask,
                    step_index=step_index,
                    action=0,
                    reward=float(reward),
                    done=done,
                    old_log_prob=0.0,
                    old_value=0.0,
                    planner_mode=int(current_mode),
                    compact_streak=int(current_compact_streak),
                    low_level_train=False,
                )
            )
            obs = next_obs
            break

        low_obs = _observation_with_compact_streak(
            _observation_without_stop_action(obs),
            current_compact_streak,
        )
        global_t, action_t, mask_t = _observation_to_tensors(low_obs, device=device)
        mode_tensor = torch.as_tensor([current_mode], device=device, dtype=torch.long)
        action_bias = _planner_logit_bias_from_action_features(action_t, mode_tensor, config)
        with torch.inference_mode():
            dist, value = model(global_t, action_t, mask_t, action_logit_bias=action_bias)
            probs = dist.probs.squeeze(0).detach().cpu().numpy()
            prob_sum = float(np.sum(probs))
            if prob_sum <= 0.0 or not np.isfinite(prob_sum):
                raise RuntimeError("planner-controlled action probabilities became non-finite or non-positive")
            probs = probs / prob_sum
            action = int(rng.choice(probs.shape[0], p=probs))
            action_tensor = torch.as_tensor([action], device=device, dtype=torch.int64)
            old_log_prob = float(dist.log_prob(action_tensor).item())
        if action == 0:
            raise RuntimeError("planner-controlled low-level policy selected STOP despite STOP being masked")

        old_value = float(value.item())
        packed_mask = _pack_mask(np.asarray(obs["membership_mask"], dtype=np.uint8))
        step_index = int(obs["step_index"])
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        total_reward += float(reward)
        steps.append(
            EpisodeStep(
                packed_membership_mask=packed_mask,
                step_index=step_index,
                action=action,
                reward=float(reward),
                done=done,
                old_log_prob=old_log_prob,
                old_value=old_value,
                planner_mode=int(current_mode),
                compact_streak=int(current_compact_streak),
                low_level_train=True,
            )
        )
        obs = next_obs
        steps_since_planner += 1
        if done:
            break

    return EpisodeTrajectory(
        episode_slot=-1,
        steps=tuple(steps),
        total_reward=float(total_reward),
        planner_steps=tuple(planner_steps),
    )


def _compute_full_grpo_episode_scores(
    trajectories: list["EpisodeTrajectory"],
    episode_contexts: list[EpisodeContext],
    config: PPOTrainingConfig,
) -> np.ndarray:
    return _grpo_objective._compute_full_grpo_episode_scores(
        trajectories,
        episode_contexts,
        config,
        final_membership_mask_fn=_final_membership_mask_from_trajectory,
        evidence_state_fn=_full_grpo_evidence_state,
        key_node_score_fn=_full_grpo_key_node_score,
    )


def _compute_full_grpo_episode_advantages(
    trajectories: list["EpisodeTrajectory"],
    episode_contexts: list[EpisodeContext],
    *,
    group_size: int,
    norm_epsilon: float,
    config: PPOTrainingConfig,
) -> np.ndarray:
    return _grpo_objective._compute_full_grpo_episode_advantages(
        trajectories,
        episode_contexts,
        group_size=group_size,
        norm_epsilon=norm_epsilon,
        config=config,
        score_fn=_compute_full_grpo_episode_scores,
    )


def _collect_episode_contexts(
    *,
    dataset: EpisodeDataset,
    batch_cells: int,
    max_steps_per_episode: int | None,
) -> list[EpisodeContext]:
    """Sample episode rows and load static contexts for one outer update."""
    contexts: list[EpisodeContext] = []
    max_attempts = max(int(batch_cells) * 10, 100)
    attempts = 0

    while len(contexts) < int(batch_cells) and attempts < max_attempts:
        needed = int(batch_cells) - len(contexts)
        sampled = dataset.sample_rows(needed)
        for row in sampled.itertuples(index=False):
            attempts += 1
            cell_id = str(row.cell_id)
            artifact_path = Path(str(row.artifact_path)).expanduser().resolve()
            ctx = dataset.load_episode_context(
                cell_id=cell_id,
                artifact_path=artifact_path,
                max_steps_per_episode=max_steps_per_episode,
            )
            if ctx is None:
                continue
            contexts.append(ctx)
            if len(contexts) >= int(batch_cells):
                break
    return contexts


def _expand_group_relative_contexts(
    *,
    contexts: list[EpisodeContext],
    group_size: int,
) -> list[EpisodeContext]:
    """Duplicate sampled contexts contiguously for grouped rollouts."""
    if group_size <= 1:
        raise ValueError("group_size must be > 1")
    expanded: list[EpisodeContext] = []
    for ctx in contexts:
        expanded.extend([ctx] * int(group_size))
    return expanded


def _collect_trajectories(
    *,
    contexts: list[EpisodeContext],
    model: ActorCritic,
    device: torch.device,
    rng: np.random.Generator,
    rollout_mode: str,
    n_rollout_workers: int,
    config: PPOTrainingConfig,
    env_cls: EnvFactory,
) -> list[EpisodeTrajectory]:
    """Collect one rollout trajectory per context."""
    if not contexts:
        return []

    mode = str(rollout_mode).strip().lower()
    if mode == "vectorized":
        if bool(config.planner_enabled):
            return _collect_trajectories_vectorized_planner(
                contexts=contexts,
                model=model,
                device=device,
                rng=rng,
                config=config,
                env_cls=env_cls,
            )
        return _collect_trajectories_vectorized(
            contexts=contexts,
            model=model,
            device=device,
            rng=rng,
            env_cls=env_cls,
        )
    if mode == "legacy":
        return _collect_trajectories_legacy(
            contexts=contexts,
            model=model,
            device=device,
            rng=rng,
            n_rollout_workers=n_rollout_workers,
            config=config,
            env_cls=env_cls,
        )
    raise ValueError(f"unsupported rollout mode: {rollout_mode!r}")


def _collect_trajectories_legacy(
    *,
    contexts: list[EpisodeContext],
    model: ActorCritic,
    device: torch.device,
    rng: np.random.Generator,
    n_rollout_workers: int,
    config: PPOTrainingConfig,
    env_cls: EnvFactory,
) -> list[EpisodeTrajectory]:
    """Legacy rollout path: one env per worker with optional threading."""
    worker_count = max(1, int(n_rollout_workers))
    seeds = rng.integers(low=0, high=np.iinfo(np.uint32).max, size=len(contexts), dtype=np.uint64)
    specs = [(i, contexts[i], int(seeds[i])) for i in range(len(contexts))]

    if worker_count == 1 or len(contexts) == 1:
        trajectories: list[EpisodeTrajectory] = []
        for episode_slot, ctx, seed in specs:
            trajectories.append(_rollout_worker(episode_slot, ctx, model, device, seed, config, env_cls))
        return trajectories

    trajectories = []
    max_workers = min(worker_count, len(contexts))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_rollout_worker, slot, ctx, model, device, seed, config, env_cls)
            for slot, ctx, seed in specs
        ]
        for future in futures:
            trajectories.append(future.result())

    trajectories.sort(key=lambda x: x.episode_slot)
    return trajectories


def _collect_trajectories_vectorized(
    *,
    contexts: list[EpisodeContext],
    model: ActorCritic,
    device: torch.device,
    rng: np.random.Generator,
    env_cls: EnvFactory,
) -> list[EpisodeTrajectory]:
    """Synchronous vectorized rollout: batch active envs each step."""
    n_envs = len(contexts)
    envs = [env_cls(ctx) for ctx in contexts]
    observations: list[dict[str, Any]] = []
    for env in envs:
        obs, _ = env.reset()
        observations.append(obs)

    episode_steps: list[list[EpisodeStep]] = [[] for _ in range(n_envs)]
    total_rewards = np.zeros((n_envs,), dtype=np.float64)
    active_slots = list(range(n_envs))

    while active_slots:
        active_obs = [observations[slot] for slot in active_slots]
        global_t, action_t, mask_t = _batch_observations_for_rollout(active_obs, device=device)

        with torch.inference_mode():
            dist, values = model(global_t, action_t, mask_t)
            probs = dist.probs
            if probs.ndim != 2:
                raise RuntimeError("expected batched action probabilities with shape (N, A)")
            if not torch.isfinite(probs).all():
                raise RuntimeError("action probabilities became non-finite")
            actions_t = dist.sample()
            log_probs_t = dist.log_prob(actions_t)

        actions = actions_t.detach().cpu().numpy().astype(np.int64, copy=False)
        old_log_probs = log_probs_t.detach().cpu().numpy().astype(np.float32, copy=False)
        old_values = values.detach().cpu().numpy().astype(np.float32, copy=False)

        next_active_slots: list[int] = []
        for local_idx, slot in enumerate(active_slots):
            obs = observations[slot]
            packed_mask = _pack_mask(np.asarray(obs["membership_mask"], dtype=np.uint8))
            step_index = int(obs["step_index"])
            action = int(actions[local_idx])
            old_log_prob = float(old_log_probs[local_idx])
            old_value = float(old_values[local_idx])

            next_obs, reward, terminated, truncated, _ = envs[slot].step(action)
            done = bool(terminated or truncated)
            total_rewards[slot] += float(reward)
            episode_steps[slot].append(
                EpisodeStep(
                    packed_membership_mask=packed_mask,
                    step_index=step_index,
                    action=action,
                    reward=float(reward),
                    done=done,
                    old_log_prob=old_log_prob,
                    old_value=old_value,
                )
            )
            observations[slot] = next_obs
            if not done:
                next_active_slots.append(slot)
        active_slots = next_active_slots

    return [
        EpisodeTrajectory(
            episode_slot=int(slot),
            steps=tuple(episode_steps[slot]),
            total_reward=float(total_rewards[slot]),
        )
        for slot in range(n_envs)
    ]


def _collect_trajectories_vectorized_planner(
    *,
    contexts: list[EpisodeContext],
    model: ActorCritic,
    device: torch.device,
    rng: np.random.Generator,
    config: PPOTrainingConfig,
    env_cls: EnvFactory,
) -> list[EpisodeTrajectory]:
    """Vectorized rollout with high-level planner checkpoints and low-level ADD-only actions."""
    n_envs = len(contexts)
    envs = [env_cls(ctx) for ctx in contexts]
    observations: list[dict[str, Any]] = []
    for env in envs:
        obs, _ = env.reset()
        observations.append(obs)

    episode_steps: list[list[EpisodeStep]] = [[] for _ in range(n_envs)]
    planner_steps: list[list[PlannerStep]] = [[] for _ in range(n_envs)]
    total_rewards = np.zeros((n_envs,), dtype=np.float64)
    current_modes = np.full((n_envs,), PLANNER_MODE_BALANCED, dtype=np.int64)
    compact_streaks = np.zeros((n_envs,), dtype=np.int32)
    steps_since_planner = np.full((n_envs,), int(config.planner_interval), dtype=np.int32)
    active_slots = list(range(n_envs))

    while active_slots:
        planner_slots = [
            slot
            for slot in active_slots
            if not planner_steps[slot] or int(steps_since_planner[slot]) >= int(config.planner_interval)
        ]
        if planner_slots:
            planner_global = torch.as_tensor(
                np.asarray(
                    [
                        _observation_with_compact_streak(
                            observations[slot],
                            int(compact_streaks[slot]),
                        )["global_features"]
                        for slot in planner_slots
                    ],
                    dtype=np.float32,
                ),
                device=device,
                dtype=torch.float32,
            )
            with torch.inference_mode():
                planner_dist = model.planner_distribution(planner_global)
                modes_t = planner_dist.sample()
                planner_log_probs_t = planner_dist.log_prob(modes_t)
            modes = modes_t.detach().cpu().numpy().astype(np.int64, copy=False)
            planner_log_probs = planner_log_probs_t.detach().cpu().numpy().astype(np.float32, copy=False)

            for local_idx, slot in enumerate(planner_slots):
                mode = int(modes[local_idx])
                current_modes[slot] = mode
                steps_since_planner[slot] = 0
                obs = observations[slot]
                planner_steps[slot].append(
                    PlannerStep(
                        packed_membership_mask=_pack_mask(np.asarray(obs["membership_mask"], dtype=np.uint8)),
                        step_index=int(obs["step_index"]),
                        mode=mode,
                        old_log_prob=float(planner_log_probs[local_idx]),
                        compact_streak=int(compact_streaks[slot]),
                    )
                )
                compact_streaks[slot] = (
                    int(compact_streaks[slot]) + 1 if mode == PLANNER_MODE_COMPACT else 0
                )

        add_slots: list[int] = []
        next_active_slots: list[int] = []
        for slot in active_slots:
            obs = observations[slot]
            has_add = bool(np.any(np.asarray(obs["action_mask"], dtype=bool)[1:]))
            if int(current_modes[slot]) == PLANNER_MODE_STOP or not has_add:
                packed_mask = _pack_mask(np.asarray(obs["membership_mask"], dtype=np.uint8))
                step_index = int(obs["step_index"])
                next_obs, reward, terminated, truncated, _ = envs[slot].step(0)
                done = bool(terminated or truncated)
                total_rewards[slot] += float(reward)
                episode_steps[slot].append(
                    EpisodeStep(
                        packed_membership_mask=packed_mask,
                        step_index=step_index,
                        action=0,
                        reward=float(reward),
                        done=done,
                        old_log_prob=0.0,
                        old_value=0.0,
                        planner_mode=int(current_modes[slot]),
                        compact_streak=int(compact_streaks[slot]),
                        low_level_train=False,
                    )
                )
                observations[slot] = next_obs
                if not done:
                    next_active_slots.append(slot)
            else:
                add_slots.append(slot)

        if add_slots:
            low_obs = [
                _observation_with_compact_streak(
                    _observation_without_stop_action(observations[slot]),
                    int(compact_streaks[slot]),
                )
                for slot in add_slots
            ]
            global_t, action_t, mask_t = _batch_observations_for_rollout(low_obs, device=device)
            mode_t = torch.as_tensor(current_modes[add_slots], device=device, dtype=torch.long)
            action_bias = _planner_logit_bias_from_action_features(action_t, mode_t, config)
            with torch.inference_mode():
                dist, values = model(global_t, action_t, mask_t, action_logit_bias=action_bias)
                actions_t = dist.sample()
                log_probs_t = dist.log_prob(actions_t)

            actions = actions_t.detach().cpu().numpy().astype(np.int64, copy=False)
            old_log_probs = log_probs_t.detach().cpu().numpy().astype(np.float32, copy=False)
            old_values = values.detach().cpu().numpy().astype(np.float32, copy=False)

            for local_idx, slot in enumerate(add_slots):
                obs = observations[slot]
                action = int(actions[local_idx])
                if action == 0:
                    raise RuntimeError("planner-controlled vectorized policy selected STOP despite STOP being masked")
                packed_mask = _pack_mask(np.asarray(obs["membership_mask"], dtype=np.uint8))
                step_index = int(obs["step_index"])
                next_obs, reward, terminated, truncated, _ = envs[slot].step(action)
                done = bool(terminated or truncated)
                total_rewards[slot] += float(reward)
                episode_steps[slot].append(
                    EpisodeStep(
                        packed_membership_mask=packed_mask,
                        step_index=step_index,
                        action=action,
                        reward=float(reward),
                        done=done,
                        old_log_prob=float(old_log_probs[local_idx]),
                        old_value=float(old_values[local_idx]),
                        planner_mode=int(current_modes[slot]),
                        compact_streak=int(compact_streaks[slot]),
                        low_level_train=True,
                    )
                )
                observations[slot] = next_obs
                steps_since_planner[slot] += 1
                if not done:
                    next_active_slots.append(slot)

        active_slots = next_active_slots

    return [
        EpisodeTrajectory(
            episode_slot=int(slot),
            steps=tuple(episode_steps[slot]),
            total_reward=float(total_rewards[slot]),
            planner_steps=tuple(planner_steps[slot]),
        )
        for slot in range(n_envs)
    ]


def _rollout_worker(
    episode_slot: int,
    ctx: EpisodeContext,
    model: ActorCritic,
    device: torch.device,
    seed: int,
    config: PPOTrainingConfig,
    env_cls: EnvFactory,
) -> EpisodeTrajectory:
    """Run one episode rollout in a worker thread."""
    local_rng = np.random.default_rng(seed)
    env = env_cls(ctx)
    traj = run_episode(env=env, model=model, device=device, rng=local_rng, config=config)
    return EpisodeTrajectory(
        episode_slot=int(episode_slot),
        steps=traj.steps,
        total_reward=float(traj.total_reward),
        planner_steps=traj.planner_steps,
    )

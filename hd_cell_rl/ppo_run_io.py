"""Run metadata, checkpoint, and log-file helpers for PPO training."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any
from zoneinfo import ZoneInfo

import torch
import yaml

from .ppo_config import PPOTrainingConfig
from .ppo_model import ActorCritic

_LOCAL_TIMEZONE = ZoneInfo("America/Chicago")
_LOCAL_TIMEZONE_NAME = "America/Chicago"


def _save_checkpoint(
    *,
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    update_index: int,
    best_moving_avg_reward: float | None,
    config: PPOTrainingConfig,
) -> None:
    payload = {
        "update_index": int(update_index),
        "best_moving_avg_reward": best_moving_avg_reward,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.to_serializable_dict(),
    }
    torch.save(payload, path)


def _build_metadata(run_dir: Path, seed: int | None) -> dict[str, Any]:
    now_utc, now_local = _now_utc_and_local()
    return {
        "created_at_utc": now_utc.isoformat(),
        "created_at_local": now_local.isoformat(),
        "local_timezone": _LOCAL_TIMEZONE_NAME,
        "run_dir": str(run_dir),
        "python_version": sys.version,
        "platform": platform.platform(),
        "seed": seed,
        "git_commit": _git_commit_hash(),
    }


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_")
    return value.lower() or "run"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _append_step_log(path: Path, event: str, payload: dict[str, Any]) -> None:
    now_utc, now_local = _now_utc_and_local()
    row = {
        "timestamp_utc": now_utc.isoformat(),
        "timestamp_local": now_local.isoformat(),
        "event": event,
        "payload": payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row))
        handle.write("\n")


def _now_utc_and_local() -> tuple[dt.datetime, dt.datetime]:
    """Return current UTC and America/Chicago timestamps."""
    now_utc = dt.datetime.now(dt.timezone.utc)
    return now_utc, now_utc.astimezone(_LOCAL_TIMEZONE)

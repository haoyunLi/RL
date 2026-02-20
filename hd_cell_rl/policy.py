"""Policy interfaces and simple baseline policies."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .actions import Action


class Policy(Protocol):
    """Policy interface that maps observation + valid actions to one action."""

    def select_action(self, observation: dict[str, Any], valid_actions: list[Action]) -> Action:
        """Pick one action to execute in the environment."""


class RandomPolicy:
    """Simple debug policy used to verify environment transitions."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def select_action(self, observation: dict[str, Any], valid_actions: list[Action]) -> Action:
        del observation
        if not valid_actions:
            raise ValueError("valid_actions must not be empty")

        index = int(self._rng.integers(low=0, high=len(valid_actions)))
        return valid_actions[index]

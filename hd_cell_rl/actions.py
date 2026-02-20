"""Action primitives for the HD spatial-cell RL environment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionType(str, Enum):
    """Discrete action family for bin assignment."""

    ADD_BIN = "add_bin"
    REMOVE_BIN = "remove_bin"
    STOP = "stop"


@dataclass(frozen=True)
class Action:
    """Concrete action with optional target bin ID."""

    kind: ActionType
    bin_id: str | None = None

    @staticmethod
    def add(bin_id: str) -> "Action":
        """Factory helper for ADD_BIN actions."""
        return Action(kind=ActionType.ADD_BIN, bin_id=bin_id)

    @staticmethod
    def remove(bin_id: str) -> "Action":
        """Factory helper for REMOVE_BIN actions."""
        return Action(kind=ActionType.REMOVE_BIN, bin_id=bin_id)

    @staticmethod
    def stop() -> "Action":
        """Factory helper for STOP actions."""
        return Action(kind=ActionType.STOP, bin_id=None)

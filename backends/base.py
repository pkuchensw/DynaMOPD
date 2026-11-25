"""Controller-facing interface. All receipt timestamps use the executor's one clock."""
from __future__ import annotations

from typing import Protocol, TypedDict

from ..cluster import GroupPlacement
from ..scheduler.protocol import Action


class Receipt(TypedDict):
    start: float
    end: float
    free_bytes: int
    value: dict


class Executor(Protocol):
    """Construct with the job config. validate_checkpoint is used only for resumable engines."""
    placements: tuple[GroupPlacement, ...]
    supervision: frozenset[str]
    supports_resume: bool
    supports_periodic_checkpoint: bool

    @staticmethod
    def validate_checkpoint(path) -> None: ...

    def now(self) -> float: ...
    def submit(self, action: Action) -> None: ...
    def wait(self) -> list[tuple[Action, Receipt]]: ...
    def close(self, failed: bool = False) -> None: ...

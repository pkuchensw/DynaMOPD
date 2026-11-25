"""Policy-facing observations. The controller settles leases between successive yields."""
from __future__ import annotations

from collections.abc import Iterable
from types import MappingProxyType
from typing import Protocol

from .protocol import Action, Group, Row


class PlanningState:
    """Borrow group/row observations for this decision only. Submit changes as actions."""

    def __init__(self, controller):
        self._controller = controller
        self.cfg = MappingProxyType(controller.cfg)
        self.groups = tuple(controller.groups)
        self.placements = tuple(controller.backend.placements)
        self.rows = MappingProxyType(controller.rows)
        self.queued_work = MappingProxyType(controller.next_work)
        self.holds = MappingProxyType(controller.holds)
        self.rejected_moves = MappingProxyType(controller.rejected_moves)
        self.version = controller.version
        self.cost = controller.cost
        self.tp_model = getattr(controller.backend, "tail_tp", None)

    @property
    def tp_pair(self):
        return self._controller.tp_pair

    @property
    def tp_prefix_tokens(self):
        return self._controller.tp_prefix_tokens

    @property
    def trainer(self):
        return self._controller.trainer

    @property
    def required_trainer(self):
        return self._controller.required_trainer

    @property
    def training_groups(self):
        return frozenset(self._controller.training_groups)

    def now(self) -> float:
        return self._controller.backend.now()

    def record_search(self, **fields):
        self._controller.emit("schedule_search", decision_id=self._controller.decision_id, **fields)

    def record_tp_candidate(self, **fields):
        self._controller.emit("tp_candidate",decision_id=self._controller.decision_id,**fields)

    def target_role(self, group: Group) -> str:
        return self._controller.target_role(group)

    def remaining(self, row: Row) -> float:
        return self._controller.remaining(row)

    def teacher_switch_s(self, group: Group, role: str) -> float:
        if role == group.role:
            return 0.
        if role in group.teacher_activation_s:
            # A cold TP mesh starts even when its CPU supervisor is prepared.
            return self.cost.values.get((group.role, role), group.teacher_activation_s[role])
        if role in group.cached_teacher_weights:
            return self.cost.get("teacher_wake")
        if group.prepared_teachers is not None and role not in group.prepared_teachers:
            return max(group.teacher_start_s, self.cost.switch(group.role, role))
        return self.cost.switch(group.role, role)

    def transfer_penalty(self, src: int, dst: int) -> float:
        if self.placements[src].node == self.placements[dst].node:
            return 0.
        return self.cfg["costs"].get("cross_node_move_s", 0.)


class SchedulingPolicy(Protocol):
    def plan(self, state: PlanningState) -> Iterable[Action]: ...

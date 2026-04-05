"""Typed schema for timing span records."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StoredSpan:
    """A single profiling span from a training iteration.

    The optional ``attributes`` dict carries scope context from the training
    loop, e.g. ``{"group_idx": 3}`` to link a ``policy_sample`` span to
    the rollout it produced.
    """

    name: str
    duration: float
    wall_start: float
    wall_end: float
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "duration": self.duration,
            "wall_start": self.wall_start,
            "wall_end": self.wall_end,
        }
        if self.attributes:
            d["attributes"] = self.attributes
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StoredSpan":
        return cls(
            name=d["name"],
            duration=d["duration"],
            wall_start=d["wall_start"],
            wall_end=d["wall_end"],
            attributes=d.get("attributes", {}),
        )

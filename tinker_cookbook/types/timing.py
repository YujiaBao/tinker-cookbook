"""Typed schema for timing span records."""

from dataclasses import dataclass
from typing import Any


@dataclass
class StoredSpan:
    """A single profiling span from a training iteration."""

    name: str
    duration: float
    wall_start: float
    wall_end: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration": self.duration,
            "wall_start": self.wall_start,
            "wall_end": self.wall_end,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StoredSpan":
        return cls(
            name=d["name"],
            duration=d["duration"],
            wall_start=d["wall_start"],
            wall_end=d["wall_end"],
        )

"""Typed schema for benchmark evaluation results."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkResult:
    """Aggregated result from running one benchmark."""

    name: str
    score: float
    num_examples: int
    num_correct: int
    num_errors: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    time_seconds: float = 0.0
    pass_at_k: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "score": self.score,
            "num_examples": self.num_examples,
            "num_correct": self.num_correct,
            "num_errors": self.num_errors,
            "metrics": dict(self.metrics),
            "time_seconds": self.time_seconds,
        }
        if self.pass_at_k:
            d["pass_at_k"] = {str(k): v for k, v in self.pass_at_k.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BenchmarkResult":
        pass_at_k_raw = d.get("pass_at_k", {})
        return cls(
            name=d["name"],
            score=d["score"],
            num_examples=d["num_examples"],
            num_correct=d["num_correct"],
            num_errors=d.get("num_errors", 0),
            metrics=d.get("metrics", {}),
            time_seconds=d.get("time_seconds", 0.0),
            pass_at_k={int(k): v for k, v in pass_at_k_raw.items()} if pass_at_k_raw else {},
        )

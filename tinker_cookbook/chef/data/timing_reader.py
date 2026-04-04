"""Incremental reader for timing_spans.jsonl files.

The timing_spans.jsonl format from trace.py's IterationWindow.write_spans_jsonl is::

    {"step": N, "spans": [{"name": "...", "duration": 1.5, "wall_start": 0.0, "wall_end": 1.5}, ...]}

Each line represents one training iteration. Spans within a step can overlap,
showing concurrent operations (e.g., async sampling + environment steps).
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TimingReader:
    """Reads timing_spans.jsonl incrementally with offset tracking.

    Each line contains ``{"step": N, "spans": [...]}`` where spans have
    ``name``, ``duration``, ``wall_start``, and ``wall_end`` fields.
    Overlapping ``wall_start``/``wall_end`` ranges indicate concurrent operations.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._offset: int = 0
        self._records: list[dict[str, Any]] = []

    @property
    def path(self) -> Path:
        return self._path

    @property
    def records(self) -> list[dict[str, Any]]:
        return self._records

    def read(self) -> list[dict[str, Any]]:
        """Read new lines since last call. Returns only new records.

        Each record is ``{"step": int, "spans": list}``.
        Also supports flat span records (one span per line) for backwards compat.
        """
        if not self._path.exists():
            return []

        file_size = self._path.stat().st_size
        if file_size <= self._offset:
            return []

        new_records: list[dict[str, Any]] = []
        with open(self._path, "r") as f:
            f.seek(self._offset)
            raw = f.read()

        if not raw.endswith("\n"):
            last_newline = raw.rfind("\n")
            if last_newline == -1:
                return []
            raw = raw[: last_newline + 1]

        self._offset += len(raw.encode("utf-8"))

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                new_records.append(record)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed timing line: %s", line[:100])

        self._records.extend(new_records)
        return new_records

    def get_spans_for_step(self, step: int) -> list[dict[str, Any]]:
        """Return all span records for a given training step.

        Handles both nested format ({"step": N, "spans": [...]}) and
        flat format ({"step": N, "name": ..., "wall_start": ..., "wall_end": ...}).
        """
        spans: list[dict[str, Any]] = []
        for record in self._records:
            if record.get("step") != step:
                continue
            if "spans" in record:
                # Nested format from write_spans_jsonl
                spans.extend(record["spans"])
            else:
                # Flat format (backwards compat)
                spans.append(record)
        return spans

    def get_all_spans_flat(self) -> list[dict[str, Any]]:
        """Flatten all records into individual spans with step annotation.

        Returns a list of ``{"step": int, "name": str, "duration": float,
        "wall_start": float, "wall_end": float}`` dicts.
        """
        flat: list[dict[str, Any]] = []
        for record in self._records:
            step = record.get("step", 0)
            if "spans" in record:
                for span in record["spans"]:
                    flat.append({"step": step, **span})
            else:
                flat.append(record)
        return flat

    def get_concurrency_analysis(self, step: int) -> dict[str, Any]:
        """Analyze concurrency for a specific step.

        Returns overlap information showing which spans ran in parallel.
        This is the key visualization data for understanding Tinker's
        async Future-based execution model.
        """
        spans = self.get_spans_for_step(step)
        if not spans:
            return {"step": step, "spans": [], "max_concurrency": 0, "timeline": []}

        # Sort by wall_start
        sorted_spans = sorted(spans, key=lambda s: s.get("wall_start", 0))

        # Find max concurrency at any point
        events: list[tuple[float, int]] = []
        for span in sorted_spans:
            ws = span.get("wall_start", 0)
            we = span.get("wall_end", ws + span.get("duration", 0))
            events.append((ws, 1))   # start
            events.append((we, -1))  # end
        events.sort(key=lambda e: (e[0], e[1]))

        max_concurrency = 0
        current = 0
        timeline: list[dict[str, Any]] = []
        for t, delta in events:
            current += delta
            max_concurrency = max(max_concurrency, current)
            timeline.append({"time": t, "concurrency": current})

        return {
            "step": step,
            "spans": sorted_spans,
            "max_concurrency": max_concurrency,
            "timeline": timeline,
        }

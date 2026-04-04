"""Incremental reader for timing_spans.jsonl files.

The timing_spans.jsonl format from trace.py's IterationWindow.write_spans_jsonl is::

    {"step": N, "spans": [{"name": "...", "duration": 1.5, "wall_start": 0.0, "wall_end": 1.5}, ...]}

Each line represents one training iteration. Spans within a step can overlap,
showing concurrent operations (e.g., async sampling + environment steps).
"""

from pathlib import Path
from typing import Any

from tinker_cookbook.chef.data.io import IncrementalJsonlReader


class TimingReader(IncrementalJsonlReader):
    """Reads timing_spans.jsonl incrementally with offset tracking.

    Each line contains ``{"step": N, "spans": [...]}`` where spans have
    ``name``, ``duration``, ``wall_start``, and ``wall_end`` fields.
    Overlapping ``wall_start``/``wall_end`` ranges indicate concurrent operations.
    """

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
                spans.extend(record["spans"])
            else:
                spans.append(record)
        return spans

    def get_all_spans_flat(self) -> list[dict[str, Any]]:
        """Flatten all records into individual spans with step annotation."""
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

        sorted_spans = sorted(spans, key=lambda s: s.get("wall_start", 0))

        events: list[tuple[float, int]] = []
        for span in sorted_spans:
            ws = span.get("wall_start", 0)
            we = span.get("wall_end", ws + span.get("duration", 0))
            events.append((ws, 1))
            events.append((we, -1))
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

"""Incremental reader for timing_spans.jsonl files."""

from typing import Any

from tinker_cookbook.chef.data.io import IncrementalJsonlReader
from tinker_cookbook.storage import Storage


class TimingReader(IncrementalJsonlReader):
    """Reads timing_spans.jsonl incrementally with offset tracking."""

    def __init__(self, storage: Storage, path: str) -> None:
        super().__init__(storage, path)

    def get_spans_for_step(self, step: int) -> list[dict[str, Any]]:
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

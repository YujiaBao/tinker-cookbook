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

    def build_span_tree(self, step: int) -> dict[str, Any]:
        """Build a hierarchical span tree from flat spans using time containment.

        A span B is a child of span A if B's time range is fully contained
        within A's time range (B.wall_start >= A.wall_start and B.wall_end <= A.wall_end).
        """
        spans = self.get_spans_for_step(step)
        if not spans:
            return {"step": step, "root": None}

        # Sort by start time, then by duration descending (longer spans first = parents)
        sorted_spans = sorted(spans, key=lambda s: (s.get("wall_start", 0), -s.get("duration", 0)))

        # Build tree nodes
        nodes: list[dict[str, Any]] = []
        for s in sorted_spans:
            nodes.append({
                "name": s.get("name", "?"),
                "duration": s.get("duration", 0),
                "wall_start": s.get("wall_start", 0),
                "wall_end": s.get("wall_end", 0),
                "attributes": s.get("attributes", {}),
                "children": [],
            })

        # Build tree using containment: B is child of A if
        # A.wall_start <= B.wall_start and B.wall_end <= A.wall_end + epsilon
        # Use a stack of open parents. A span is a child of the deepest
        # parent that fully contains it. When we see a span that starts
        # after the current parent ends, pop until we find a valid parent.
        EPS = 0.01  # tolerance for floating-point wall-clock comparisons
        root_children: list[dict[str, Any]] = []
        stack: list[dict[str, Any]] = []

        for node in nodes:
            # Pop parents that have ended before this node starts
            while stack and stack[-1]["wall_end"] + EPS < node["wall_start"]:
                stack.pop()

            # Check if the top of stack truly contains this node
            if stack and node["wall_end"] <= stack[-1]["wall_end"] + EPS:
                stack[-1]["children"].append(node)
            else:
                # Not contained — this is a root-level sibling
                # Pop any remaining stack entries that don't contain us
                while stack and node["wall_end"] > stack[-1]["wall_end"] + EPS:
                    stack.pop()
                if stack:
                    stack[-1]["children"].append(node)
                else:
                    root_children.append(node)

            stack.append(node)

        # Compute total wall time for the iteration
        all_starts = [n["wall_start"] for n in nodes]
        all_ends = [n["wall_end"] for n in nodes]
        total_duration = max(all_ends) - min(all_starts) if all_starts else 0

        return {
            "step": step,
            "total_duration": total_duration,
            "root": {
                "name": "iteration",
                "duration": total_duration,
                "wall_start": min(all_starts) if all_starts else 0,
                "wall_end": max(all_ends) if all_ends else 0,
                "attributes": {},
                "children": root_children,
            },
        }

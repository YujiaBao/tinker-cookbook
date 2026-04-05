import { useEffect, useMemo, useState } from 'react';
import {
  Bar, BarChart, CartesianGrid, Cell,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { TimingTree } from './TimingTree';

interface Props {
  runId: string;
}

interface FlatSpan {
  step: number;
  name: string;
  duration: number;
  wall_start: number;
  wall_end: number;
  attributes?: Record<string, unknown>;
}

// Colors that work on both light and dark themes
const SPAN_PALETTE = [
  '#8bbe3a', '#a78bfa', '#e5a11c', '#e85850', '#6aad7a',
  '#ec4899', '#06b6d4', '#f97316', '#64748b', '#14b8a6',
];

function getColor(name: string, names: string[]): string {
  const idx = names.indexOf(name);
  return SPAN_PALETTE[idx % SPAN_PALETTE.length];
}

export function TimingPanel({ runId }: Props) {
  const [spans, setSpans] = useState<FlatSpan[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [hoveredSpan, setHoveredSpan] = useState<FlatSpan | null>(null);
  const [hiddenSpanNames, setHiddenSpanNames] = useState<Set<string>>(new Set());
  const [minDuration, setMinDuration] = useState<string>('');
  const [treeData, setTreeData] = useState<{ root: any; total_duration: number } | null>(null);

  useEffect(() => {
    fetch(`/api/runs/${runId}/timing/flat`)
      .then((r) => r.json())
      .then((data) => setSpans(data.spans ?? []))
      .catch(() => setSpans([]))
      .finally(() => setLoading(false));
  }, [runId]);

  // Filter spans by visibility and min duration
  const filteredSpans = useMemo(() => {
    const minDur = minDuration !== '' ? Number(minDuration) : 0;
    return spans.filter((s) => !hiddenSpanNames.has(s.name) && s.duration >= minDur);
  }, [spans, hiddenSpanNames, minDuration]);

  const steps = useMemo(() => [...new Set(filteredSpans.map((s) => s.step))].sort((a, b) => a - b), [filteredSpans]);
  const allSpanNames = useMemo(() => [...new Set(spans.map((s) => s.name))].sort(), [spans]);
  const spanNames = useMemo(() => [...new Set(filteredSpans.map((s) => s.name))].sort(), [filteredSpans]);

  // Per-step total duration for the overview chart
  const stepDurations = useMemo(() => {
    const map = new Map<number, Record<string, number>>();
    for (const s of filteredSpans) {
      if (!map.has(s.step)) map.set(s.step, { step: s.step });
      const row = map.get(s.step)!;
      row[s.name] = (row[s.name] || 0) + s.duration;
    }
    return Array.from(map.values()).sort((a, b) => (a.step as number) - (b.step as number));
  }, [spans]);

  // Per-step wall-clock duration (from first span start to last span end)
  const wallDurations = useMemo(() => {
    const map = new Map<number, { step: number; wall: number; sum: number }>();
    for (const s of filteredSpans) {
      const existing = map.get(s.step);
      if (!existing) {
        map.set(s.step, { step: s.step, wall: s.wall_end - s.wall_start, sum: s.duration });
      } else {
        existing.wall = Math.max(existing.wall, s.wall_end) - Math.min(0, s.wall_start);
        existing.sum += s.duration;
      }
    }
    // Recalculate wall properly
    for (const step of map.keys()) {
      const stepSpans = spans.filter((s) => s.step === step);
      const minW = Math.min(...stepSpans.map((s) => s.wall_start));
      const maxW = Math.max(...stepSpans.map((s) => s.wall_end));
      map.get(step)!.wall = maxW - minW;
    }
    return Array.from(map.values()).sort((a, b) => a.step - b.step);
  }, [spans]);

  const displayStep = selectedStep ?? steps[0] ?? 0;

  // Fetch tree data for selected step
  useEffect(() => {
    if (steps.length === 0) return;
    fetch(`/api/runs/${runId}/timing/tree/${displayStep}`)
      .then((r) => r.json())
      .then((data) => setTreeData(data))
      .catch(() => setTreeData(null));
  }, [runId, displayStep, steps.length]);
  const stepSpans = useMemo(
    () => filteredSpans.filter((s) => s.step === displayStep).sort((a, b) => a.wall_start - b.wall_start),
    [filteredSpans, displayStep]
  );

  if (loading) return <div className="loading">Loading timing data...</div>;
  if (spans.length === 0) return <div className="empty-state">No timing data available</div>;

  return (
    <div>
      {/* Per-step wall time trend */}
      <div className="card" style={{ marginBottom: '0.75rem' }}>
        <div className="card-header">
          <span className="card-title">Wall Time per Step</span>
          <span className="text-muted" style={{ fontSize: '0.625rem', fontWeight: 400, textTransform: 'none', letterSpacing: 0 }}>
            Click a bar to inspect that step's waterfall
          </span>
        </div>
        <ResponsiveContainer width="100%" height={160}>
          <BarChart data={wallDurations}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="step" stroke="var(--text-muted)" tick={{ fontSize: 10 }} />
            <YAxis stroke="var(--text-muted)" tick={{ fontSize: 10 }} unit="s" width={45} />
            <Tooltip
              contentStyle={{ background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: 6, fontSize: '0.75rem' }}
              formatter={(value: unknown) => [`${Number(value).toFixed(3)}s`]}
            />
            <Bar dataKey="wall" fill="var(--accent)" radius={[2, 2, 0, 0]} cursor="pointer"
              onClick={(_data: unknown, index: number) => setSelectedStep(wallDurations[index]?.step ?? 0)}
            >
              {wallDurations.map((entry) => (
                <Cell
                  key={entry.step}
                  fill={entry.step === displayStep ? 'var(--accent)' : 'var(--border-bright)'}
                  opacity={entry.step === displayStep ? 1 : 0.6}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Stacked duration breakdown over steps */}
      {stepDurations.length > 1 && (
        <div className="card" style={{ marginBottom: '0.75rem' }}>
          <div className="card-header">
            <span className="card-title">Duration Breakdown by Span</span>
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={stepDurations}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="step" stroke="var(--text-muted)" tick={{ fontSize: 10 }} />
              <YAxis stroke="var(--text-muted)" tick={{ fontSize: 10 }} unit="s" width={45} />
              <Tooltip
                contentStyle={{ background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: 6, fontSize: '0.7rem' }}
                formatter={(value: unknown, name: unknown) => [`${Number(value).toFixed(3)}s`, String(name)]}
              />
              {spanNames.map((name) => (
                <Bar key={name} dataKey={name} stackId="a" fill={getColor(name, spanNames)} />
              ))}
            </BarChart>
          </ResponsiveContainer>
          {/* Legend */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.375rem 0.75rem', marginTop: '0.375rem' }}>
            {spanNames.map((name) => (
              <span key={name} style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.625rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                <span style={{ width: 8, height: 8, borderRadius: 2, background: getColor(name, spanNames), flexShrink: 0 }} />
                {name}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Step selector */}
      <div className="filters-bar">
        <div className="filter-group" style={{ flex: '1 1 auto' }}>
          <span className="filter-label">Step</span>
          <button className="theme-toggle" style={{ padding: '0.1875rem 0.375rem' }}
            onClick={() => { const i = steps.indexOf(displayStep); if (i > 0) setSelectedStep(steps[i - 1]); }}
            disabled={displayStep === steps[0]}
          >Prev</button>
          <input type="range" min={0} max={steps.length - 1}
            value={steps.indexOf(displayStep)}
            onChange={(e) => setSelectedStep(steps[Number(e.target.value)])}
            style={{ flex: 1, minWidth: '80px', accentColor: 'var(--accent)', cursor: 'pointer' }}
          />
          <button className="theme-toggle" style={{ padding: '0.1875rem 0.375rem' }}
            onClick={() => { const i = steps.indexOf(displayStep); if (i < steps.length - 1) setSelectedStep(steps[i + 1]); }}
            disabled={displayStep === steps[steps.length - 1]}
          >Next</button>
          <span className="mono" style={{ fontSize: '0.75rem', minWidth: '40px' }}>{displayStep}</span>
        </div>
        <div className="filter-group">
          <span className="filter-label">Min dur</span>
          <input
            type="number"
            placeholder="0"
            value={minDuration}
            onChange={(e) => setMinDuration(e.target.value)}
            step="0.01"
            style={{ width: '55px' }}
          />
          <span style={{ fontSize: '0.5625rem', color: 'var(--text-muted)' }}>s</span>
        </div>
        <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', marginLeft: 'auto' }}>
          {stepSpans.length} spans
          {hiddenSpanNames.size > 0 && ` (${hiddenSpanNames.size} hidden)`}
          {stepSpans.length > 0 && (() => {
            const minW = Math.min(...stepSpans.map((s) => s.wall_start));
            const maxW = Math.max(...stepSpans.map((s) => s.wall_end));
            return ` · ${(maxW - minW).toFixed(2)}s wall`;
          })()}
        </div>
      </div>

      {/* Hierarchical span tree — primary view */}
      {treeData?.root && (
        <div className="card" style={{ marginBottom: '0.75rem' }}>
          <div className="card-header">
            <span className="card-title">Step {displayStep} — Call Hierarchy</span>
            <span className="text-muted" style={{ fontSize: '0.625rem', fontWeight: 400, textTransform: 'none', letterSpacing: 0 }}>
              {treeData.total_duration.toFixed(2)}s total
            </span>
          </div>
          <TimingTree root={treeData.root} totalDuration={treeData.total_duration} runId={runId} step={displayStep} />
        </div>
      )}

      {/* Flat waterfall — secondary detailed view */}
      {stepSpans.length > 0 && (
        <div className="card" style={{ marginBottom: '0.75rem' }}>
          <div className="card-header">
            <span className="card-title">Step {displayStep} — Waterfall</span>
          </div>
          <InteractiveWaterfall
            spans={stepSpans}
            spanNames={spanNames}
            hoveredSpan={hoveredSpan}
            onHover={setHoveredSpan}
          />
        </div>
      )}

      {/* Span detail on hover */}
      {hoveredSpan && (
        <div className="card" style={{ marginBottom: '0.75rem', borderColor: getColor(hoveredSpan.name, spanNames) }}>
          <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap', fontSize: '0.8125rem', alignItems: 'flex-start' }}>
            <div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.625rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Span</div>
              <div className="mono" style={{ fontWeight: 700, color: getColor(hoveredSpan.name, spanNames) }}>{hoveredSpan.name}</div>
            </div>
            <div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.625rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Duration</div>
              <div className="mono" style={{ fontWeight: 600 }}>{hoveredSpan.duration.toFixed(4)}s</div>
            </div>
            <div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.625rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Wall Start</div>
              <div className="mono">{hoveredSpan.wall_start.toFixed(4)}s</div>
            </div>
            <div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.625rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Wall End</div>
              <div className="mono">{hoveredSpan.wall_end.toFixed(4)}s</div>
            </div>
            {hoveredSpan.attributes?.group_idx != null && (
              <div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.625rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Rollout</div>
                <a
                  href={`/runs/${runId}/iterations/${displayStep}/rollouts/${hoveredSpan.attributes.group_idx}/0`}
                  style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8125rem', fontWeight: 600 }}
                >
                  Group {String(hoveredSpan.attributes.group_idx)} →
                </a>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Summary table */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">Aggregate Statistics</span>
          <span className="text-muted" style={{ fontSize: '0.625rem', fontWeight: 400, textTransform: 'none', letterSpacing: 0 }}>
            Click a span name to show/hide it
          </span>
        </div>
        <table>
          <thead>
            <tr>
              <th>Span</th>
              <th>Count</th>
              <th>Total</th>
              <th>Mean</th>
              <th>Max</th>
              <th>% of Total</th>
            </tr>
          </thead>
          <tbody>
            {(() => {
              const stats = new Map<string, { total: number; count: number; max: number }>();
              let grandTotal = 0;
              for (const s of filteredSpans) {
                if (!stats.has(s.name)) stats.set(s.name, { total: 0, count: 0, max: 0 });
                const st = stats.get(s.name)!;
                st.total += s.duration;
                st.count += 1;
                st.max = Math.max(st.max, s.duration);
                grandTotal += s.duration;
              }
              return [...stats.entries()]
                .sort((a, b) => b[1].total - a[1].total)
                .map(([name, s]) => (
                  <tr key={name} style={{ cursor: 'pointer' }}
                    onClick={() => setHiddenSpanNames((prev) => {
                      const next = new Set(prev);
                      if (next.has(name)) next.delete(name);
                      else next.add(name);
                      return next;
                    })}
                  >
                    <td style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', opacity: hiddenSpanNames.has(name) ? 0.4 : 1 }}>
                      <span style={{ width: 10, height: 10, borderRadius: 2, background: getColor(name, allSpanNames), flexShrink: 0 }} />
                      <span className="mono" style={{ textDecoration: hiddenSpanNames.has(name) ? 'line-through' : 'none' }}>{name}</span>
                    </td>
                    <td className="mono">{s.count}</td>
                    <td className="mono">{s.total.toFixed(3)}s</td>
                    <td className="mono">{(s.total / s.count).toFixed(3)}s</td>
                    <td className="mono">{s.max.toFixed(3)}s</td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                        <div style={{
                          width: `${(s.total / grandTotal) * 100}%`,
                          maxWidth: '80px',
                          height: 6,
                          borderRadius: 3,
                          background: getColor(name, allSpanNames),
                          minWidth: 2,
                        }} />
                        <span className="mono text-muted" style={{ fontSize: '0.6875rem' }}>
                          {((s.total / grandTotal) * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                  </tr>
                ));
            })()}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function InteractiveWaterfall({
  spans, spanNames, hoveredSpan, onHover,
}: {
  spans: FlatSpan[];
  spanNames: string[];
  hoveredSpan: FlatSpan | null;
  onHover: (span: FlatSpan | null) => void;
}) {
  const minWall = Math.min(...spans.map((s) => s.wall_start));
  const maxWall = Math.max(...spans.map((s) => s.wall_end));
  const totalDuration = maxWall - minWall || 1;

  // Assign lanes to avoid overlap — greedy lane packing
  const lanes: { end: number }[] = [];
  const laneAssignments: number[] = [];

  for (const span of spans) {
    let assigned = false;
    for (let i = 0; i < lanes.length; i++) {
      if (span.wall_start >= lanes[i].end - 0.001) {
        lanes[i].end = span.wall_end;
        laneAssignments.push(i);
        assigned = true;
        break;
      }
    }
    if (!assigned) {
      lanes.push({ end: span.wall_end });
      laneAssignments.push(lanes.length - 1);
    }
  }

  const rowHeight = 26;
  const totalHeight = lanes.length * rowHeight + 24;

  // Time axis ticks
  const numTicks = 6;
  const ticks = Array.from({ length: numTicks + 1 }, (_, i) => ({
    time: (i / numTicks) * totalDuration,
    x: (i / numTicks) * 100,
  }));

  return (
    <div style={{ position: 'relative', overflow: 'hidden' }}>
      {/* Time grid lines */}
      <svg style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: totalHeight, pointerEvents: 'none' }}>
        {ticks.map((tick, i) => (
          <line key={i} x1={`${tick.x}%`} y1="0" x2={`${tick.x}%`} y2={totalHeight - 20}
            stroke="var(--border)" strokeWidth="0.5" strokeDasharray="4 4" />
        ))}
      </svg>

      {/* Bars */}
      <div style={{ position: 'relative', height: totalHeight - 20 }}>
        {spans.map((span, idx) => {
          const left = ((span.wall_start - minWall) / totalDuration) * 100;
          const width = ((span.wall_end - span.wall_start) / totalDuration) * 100;
          const lane = laneAssignments[idx];
          const isHovered = hoveredSpan === span;
          const color = getColor(span.name, spanNames);

          return (
            <div
              key={idx}
              onMouseEnter={() => onHover(span)}
              onMouseLeave={() => onHover(null)}
              style={{
                position: 'absolute',
                left: `${left}%`,
                width: `${Math.max(width, 0.3)}%`,
                top: lane * rowHeight,
                height: rowHeight - 4,
                borderRadius: 4,
                background: color,
                opacity: isHovered ? 1 : 0.8,
                border: isHovered ? '2px solid var(--text-primary)' : '1px solid transparent',
                boxSizing: 'border-box',
                display: 'flex',
                alignItems: 'center',
                padding: '0 6px',
                cursor: 'pointer',
                transition: 'opacity 0.1s, border 0.1s',
                overflow: 'hidden',
                whiteSpace: 'nowrap',
              }}
            >
              {width > 8 && (
                <span style={{ fontSize: '0.6rem', color: 'white', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                  {span.name}
                </span>
              )}
              {width > 15 && (
                <span style={{ fontSize: '0.55rem', color: 'rgba(255,255,255,0.7)', marginLeft: '0.375rem', fontFamily: 'var(--font-mono)' }}>
                  {span.duration.toFixed(3)}s
                </span>
              )}
            </div>
          );
        })}
      </div>

      {/* Time axis */}
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.5625rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', paddingTop: 4 }}>
        {ticks.map((tick, i) => (
          <span key={i}>{tick.time.toFixed(2)}s</span>
        ))}
      </div>
    </div>
  );
}

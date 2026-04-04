import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { TimingRecord } from '../api/types';

interface Props {
  runId: string;
}

// Color palette for span types
const SPAN_COLORS: Record<string, string> = {
  forward_backward: '#6366f1',
  optim_step: '#22c55e',
  policy_sample: '#f59e0b',
  sample_async: '#f59e0b',
  run_evals: '#06b6d4',
  gather_rollouts: '#ec4899',
  compute_advantages: '#8b5cf6',
};

function getSpanColor(name: string): string {
  return SPAN_COLORS[name] ?? '#64748b';
}

export function TimingPanel({ runId }: Props) {
  const [records, setRecords] = useState<TimingRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);

  useEffect(() => {
    api
      .getTiming(runId)
      .then((resp) => setRecords(resp.records))
      .catch(() => setRecords([]))
      .finally(() => setLoading(false));
  }, [runId]);

  if (loading) return <div className="loading">Loading timing data...</div>;
  if (records.length === 0) {
    return <div className="empty-state">No timing data available</div>;
  }

  // Get unique steps
  const steps = [...new Set(records.map((r) => r.step))].sort((a, b) => a - b);

  // Get unique span names for the legend
  const spanNames = [...new Set(records.map((r) => r.name))].sort();

  // Filter records for selected step, or show all if none selected
  const displayRecords = selectedStep !== null
    ? records.filter((r) => r.step === selectedStep)
    : records;

  // Group by step for the waterfall view
  const stepGroups = new Map<number, TimingRecord[]>();
  for (const r of displayRecords) {
    if (!stepGroups.has(r.step)) stepGroups.set(r.step, []);
    stepGroups.get(r.step)!.push(r);
  }

  // Aggregate timing stats
  const stats = new Map<string, { total: number; count: number; max: number }>();
  for (const r of records) {
    const duration = r.end_time - r.start_time;
    if (!stats.has(r.name)) stats.set(r.name, { total: 0, count: 0, max: 0 });
    const s = stats.get(r.name)!;
    s.total += duration;
    s.count += 1;
    s.max = Math.max(s.max, duration);
  }

  return (
    <div>
      {/* Timing summary table */}
      <div className="card" style={{ marginBottom: '16px' }}>
        <div className="card-header">
          <span className="card-title">Timing Summary</span>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            {records.length} spans across {steps.length} steps
          </span>
        </div>
        <table className="data-table">
          <thead>
            <tr>
              <th>Span Name</th>
              <th>Count</th>
              <th>Total (s)</th>
              <th>Mean (s)</th>
              <th>Max (s)</th>
            </tr>
          </thead>
          <tbody>
            {spanNames.map((name) => {
              const s = stats.get(name);
              if (!s) return null;
              return (
                <tr key={name} style={{ cursor: 'default' }}>
                  <td>
                    <span
                      style={{
                        display: 'inline-block',
                        width: 8,
                        height: 8,
                        borderRadius: '50%',
                        background: getSpanColor(name),
                        marginRight: 8,
                      }}
                    />
                    {name}
                  </td>
                  <td className="mono">{s.count}</td>
                  <td className="mono">{s.total.toFixed(3)}</td>
                  <td className="mono">{(s.total / s.count).toFixed(3)}</td>
                  <td className="mono">{s.max.toFixed(3)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Step selector */}
      <div className="filters-bar">
        <div className="filter-group">
          <span className="filter-label">Step</span>
          <select
            value={selectedStep ?? 'all'}
            onChange={(e) =>
              setSelectedStep(e.target.value === 'all' ? null : Number(e.target.value))
            }
          >
            <option value="all">All ({steps.length} steps)</option>
            {steps.map((step) => (
              <option key={step} value={step}>
                Step {step}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Waterfall view */}
      {Array.from(stepGroups.entries())
        .sort(([a], [b]) => a - b)
        .slice(0, 20) // Limit to 20 steps in view
        .map(([step, spans]) => {
          const minWall = Math.min(...spans.map((s) => s.wall_start));
          const maxWall = Math.max(...spans.map((s) => s.wall_end));
          const totalWidth = maxWall - minWall || 1;

          return (
            <div key={step} className="card" style={{ marginBottom: '8px', padding: '12px 16px' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '8px' }}>
                Step {step} — {totalWidth.toFixed(2)}s total
              </div>
              <div style={{ position: 'relative', height: spans.length * 24 + 4 }}>
                {spans.map((span, idx) => {
                  const left = ((span.wall_start - minWall) / totalWidth) * 100;
                  const width = ((span.wall_end - span.wall_start) / totalWidth) * 100;
                  return (
                    <div
                      key={`${span.name}-${idx}`}
                      title={`${span.name}: ${(span.end_time - span.start_time).toFixed(3)}s`}
                      style={{
                        position: 'absolute',
                        left: `${left}%`,
                        width: `${Math.max(width, 0.5)}%`,
                        top: idx * 24,
                        height: 20,
                        borderRadius: 3,
                        background: getSpanColor(span.name),
                        fontSize: '0.65rem',
                        display: 'flex',
                        alignItems: 'center',
                        padding: '0 4px',
                        color: 'white',
                        overflow: 'hidden',
                        whiteSpace: 'nowrap',
                        opacity: 0.85,
                      }}
                    >
                      {width > 8 ? span.name : ''}
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
    </div>
  );
}

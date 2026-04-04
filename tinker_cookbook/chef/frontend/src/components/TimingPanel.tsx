import { useEffect, useState } from 'react';

interface Props {
  runId: string;
}

interface FlatSpan {
  step: number;
  name: string;
  duration: number;
  wall_start: number;
  wall_end: number;
}

interface ConcurrencyData {
  step: number;
  spans: FlatSpan[];
  max_concurrency: number;
  timeline: { time: number; concurrency: number }[];
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
  env_step: '#14b8a6',
};

function getSpanColor(name: string): string {
  return SPAN_COLORS[name] ?? '#64748b';
}

export function TimingPanel({ runId }: Props) {
  const [spans, setSpans] = useState<FlatSpan[]>([]);
  const [concurrency, setConcurrency] = useState<ConcurrencyData | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);

  // Load all flat spans
  useEffect(() => {
    fetch(`/api/runs/${runId}/timing/flat`)
      .then((r) => r.json())
      .then((data) => setSpans(data.spans ?? []))
      .catch(() => setSpans([]))
      .finally(() => setLoading(false));
  }, [runId]);

  // Load concurrency analysis for selected step
  useEffect(() => {
    if (selectedStep === null) {
      setConcurrency(null);
      return;
    }
    fetch(`/api/runs/${runId}/timing/concurrency/${selectedStep}`)
      .then((r) => r.json())
      .then((data) => setConcurrency(data))
      .catch(() => setConcurrency(null));
  }, [runId, selectedStep]);

  if (loading) return <div className="loading">Loading timing data...</div>;
  if (spans.length === 0) {
    return <div className="empty-state">No timing data available</div>;
  }

  const steps = [...new Set(spans.map((s) => s.step))].sort((a, b) => a - b);
  const spanNames = [...new Set(spans.map((s) => s.name))].sort();

  // Select first step by default
  if (selectedStep === null && steps.length > 0) {
    // Don't setState in render — use the first step for display
  }
  const displayStep = selectedStep ?? steps[0];

  // Aggregate stats across all spans
  const stats = new Map<string, { total: number; count: number; max: number }>();
  for (const s of spans) {
    if (!stats.has(s.name)) stats.set(s.name, { total: 0, count: 0, max: 0 });
    const st = stats.get(s.name)!;
    st.total += s.duration;
    st.count += 1;
    st.max = Math.max(st.max, s.duration);
  }

  // Get spans for waterfall display
  const stepSpans = spans.filter((s) => s.step === displayStep);

  return (
    <div>
      {/* Timing summary */}
      <div className="card" style={{ marginBottom: '16px' }}>
        <div className="card-header">
          <span className="card-title">Timing Summary</span>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            {spans.length} spans across {steps.length} steps
          </span>
        </div>
        <table className="data-table">
          <thead>
            <tr>
              <th>Span</th>
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
                    <span style={{
                      display: 'inline-block', width: 8, height: 8,
                      borderRadius: '50%', background: getSpanColor(name), marginRight: 8,
                    }} />
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

      {/* Step selector + concurrency indicator */}
      <div className="filters-bar">
        <div className="filter-group">
          <span className="filter-label">Step</span>
          <select
            value={displayStep}
            onChange={(e) => setSelectedStep(Number(e.target.value))}
          >
            {steps.map((step) => (
              <option key={step} value={step}>Step {step}</option>
            ))}
          </select>
        </div>
        {concurrency && (
          <div style={{ fontSize: '0.8rem' }}>
            <span className="text-muted">Peak concurrency: </span>
            <span className={`badge ${concurrency.max_concurrency > 1 ? 'badge-blue' : ''}`}>
              {concurrency.max_concurrency}x parallel
            </span>
          </div>
        )}
      </div>

      {/* Waterfall for selected step */}
      {stepSpans.length > 0 && (
        <div className="card" style={{ marginBottom: '16px' }}>
          <div className="card-header">
            <span className="card-title">Step {displayStep} — Execution Waterfall</span>
          </div>
          <WaterfallChart spans={stepSpans} />
        </div>
      )}

      {/* Concurrency timeline */}
      {concurrency && concurrency.timeline.length > 0 && (
        <div className="card">
          <div className="card-header">
            <span className="card-title">Concurrency Over Time</span>
          </div>
          <ConcurrencyTimeline data={concurrency} />
        </div>
      )}
    </div>
  );
}

function WaterfallChart({ spans }: { spans: FlatSpan[] }) {
  if (spans.length === 0) return null;
  const minWall = Math.min(...spans.map((s) => s.wall_start));
  const maxWall = Math.max(...spans.map((s) => s.wall_end));
  const totalWidth = maxWall - minWall || 1;

  // Sort by start time for clean visual layout
  const sorted = [...spans].sort((a, b) => a.wall_start - b.wall_start);

  return (
    <div style={{ position: 'relative', height: sorted.length * 28 + 8, marginTop: 8 }}>
      {sorted.map((span, idx) => {
        const left = ((span.wall_start - minWall) / totalWidth) * 100;
        const width = ((span.wall_end - span.wall_start) / totalWidth) * 100;
        return (
          <div
            key={`${span.name}-${idx}`}
            title={`${span.name}: ${span.duration.toFixed(3)}s (wall: ${span.wall_start.toFixed(3)}s - ${span.wall_end.toFixed(3)}s)`}
            style={{
              position: 'absolute',
              left: `${left}%`,
              width: `${Math.max(width, 0.5)}%`,
              top: idx * 28,
              height: 22,
              borderRadius: 4,
              background: getSpanColor(span.name),
              fontSize: '0.65rem',
              display: 'flex',
              alignItems: 'center',
              padding: '0 6px',
              color: 'white',
              overflow: 'hidden',
              whiteSpace: 'nowrap',
              opacity: 0.9,
              cursor: 'default',
            }}
          >
            {width > 10 && (
              <>
                <span style={{ fontWeight: 600 }}>{span.name}</span>
                <span style={{ marginLeft: 6, opacity: 0.8 }}>{span.duration.toFixed(3)}s</span>
              </>
            )}
            {width <= 10 && width > 4 && <span style={{ fontWeight: 600 }}>{span.name}</span>}
          </div>
        );
      })}

      {/* Time axis labels */}
      <div style={{
        position: 'absolute', bottom: -16, left: 0, right: 0,
        display: 'flex', justifyContent: 'space-between',
        fontSize: '0.6rem', color: 'var(--text-muted)',
      }}>
        <span>0s</span>
        <span>{(totalWidth / 2).toFixed(2)}s</span>
        <span>{totalWidth.toFixed(2)}s</span>
      </div>
    </div>
  );
}

function ConcurrencyTimeline({ data }: { data: ConcurrencyData }) {
  const { timeline, max_concurrency } = data;
  if (timeline.length === 0) return null;

  const minTime = timeline[0].time;
  const maxTime = timeline[timeline.length - 1].time;
  const timeRange = maxTime - minTime || 1;
  const height = 80;

  // Build path for the step chart
  const points: string[] = [];
  for (const pt of timeline) {
    const x = ((pt.time - minTime) / timeRange) * 100;
    const y = height - (pt.concurrency / Math.max(max_concurrency, 1)) * (height - 10);
    if (points.length > 0) {
      // Step chart: go horizontal first, then vertical
      const prevY = points[points.length - 1].split(',')[1];
      points.push(`${x},${prevY}`);
    }
    points.push(`${x},${y}`);
  }

  return (
    <div style={{ padding: '8px 0' }}>
      <svg width="100%" height={height + 20} viewBox={`0 0 100 ${height + 20}`} preserveAspectRatio="none">
        {/* Grid lines */}
        {Array.from({ length: max_concurrency + 1 }, (_, i) => {
          const y = height - (i / Math.max(max_concurrency, 1)) * (height - 10);
          return (
            <g key={i}>
              <line x1="0" y1={y} x2="100" y2={y} stroke="var(--border)" strokeWidth="0.2" />
              <text x="1" y={y - 1} fontSize="3" fill="var(--text-muted)">{i}</text>
            </g>
          );
        })}
        {/* Concurrency line */}
        <polyline
          points={points.join(' ')}
          fill="none"
          stroke="var(--accent)"
          strokeWidth="0.5"
        />
      </svg>
      <div style={{
        display: 'flex', justifyContent: 'space-between',
        fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: 4,
      }}>
        <span>{minTime.toFixed(3)}s</span>
        <span>{maxTime.toFixed(3)}s</span>
      </div>
    </div>
  );
}

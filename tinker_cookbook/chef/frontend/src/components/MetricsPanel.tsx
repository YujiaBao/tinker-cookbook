import { useEffect, useMemo, useRef, useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api } from '../api/client';
import type { MetricRecord } from '../api/types';

const COLORS = [
  '#8bbe3a', '#a78bfa', '#e5a11c', '#e85850', '#6aad7a',
  '#ec4899', '#06b6d4', '#f97316', '#64748b', '#14b8a6',
];

interface Props {
  runId: string;
  onStepClick?: (step: number) => void;
}

const METRIC_GROUPS: [string, (key: string) => boolean][] = [
  ['Reward & Correctness', (k) => k.includes('reward') || k.includes('correct') || k.includes('format') || k.includes('by_group')],
  ['Optimization', (k) => k.startsWith('optim/')],
  ['Tokens & Episodes', (k) => k.includes('tokens') || k.includes('episodes') || k.includes('turns')],
  ['Progress', (k) => k.startsWith('progress/')],
  ['Timing', (k) => k.startsWith('time/')],
];

function groupMetricKeys(keys: string[]): Map<string, string[]> {
  const groups = new Map<string, string[]>();
  for (const key of keys) {
    if (key.endsWith(':total') || key.endsWith(':count')) continue;
    let assigned = false;
    for (const [groupName, matcher] of METRIC_GROUPS) {
      if (matcher(key)) {
        if (!groups.has(groupName)) groups.set(groupName, []);
        groups.get(groupName)!.push(key);
        assigned = true;
        break;
      }
    }
    if (!assigned) {
      const slashIdx = key.indexOf('/');
      const prefix = slashIdx > 0 ? key.substring(0, slashIdx) : 'Other';
      if (!groups.has(prefix)) groups.set(prefix, []);
      groups.get(prefix)!.push(key);
    }
  }
  return groups;
}

/** Compute EMA-smoothed data. Returns new array with smoothed values. */
function applyEMA(data: MetricRecord[], keys: string[], alpha: number): MetricRecord[] {
  if (data.length === 0) return data;
  const result: MetricRecord[] = [];
  const ema: Record<string, number> = {};

  for (const record of data) {
    const smoothed: MetricRecord = { step: record.step };
    for (const key of keys) {
      const raw = record[key];
      if (typeof raw !== 'number') continue;
      if (ema[key] === undefined) {
        ema[key] = raw;
      } else {
        ema[key] = alpha * ema[key] + (1 - alpha) * raw;
      }
      smoothed[key] = ema[key];
    }
    result.push(smoothed);
  }
  return result;
}

const EMA_OPTIONS = [
  { label: 'Off', alpha: 0 },
  { label: '0.6', alpha: 0.6 },
  { label: '0.9', alpha: 0.9 },
  { label: '0.95', alpha: 0.95 },
  { label: '0.99', alpha: 0.99 },
];

function MetricChart({ prefix, metricKeys, data, onStepClick }: {
  prefix: string;
  metricKeys: string[];
  data: MetricRecord[];
  onStepClick?: (step: number) => void;
}) {
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [emaAlpha, setEmaAlpha] = useState(0);

  const toggleKey = (key: string) => {
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const shortName = (key: string) => key.includes('/') ? key.split('/').slice(1).join('/') : key;

  const smoothedData = useMemo(
    () => emaAlpha > 0 ? applyEMA(data, metricKeys, emaAlpha) : null,
    [data, metricKeys, emaAlpha]
  );

  return (
    <div className="chart-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '0.5rem' }}>
        <div className="chart-title">{prefix}</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
          {/* EMA selector */}
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.5625rem', color: 'var(--text-muted)' }}>smooth:</span>
          <div style={{ display: 'flex', gap: '1px' }}>
            {EMA_OPTIONS.map((opt) => (
              <button
                key={opt.label}
                onClick={() => setEmaAlpha(opt.alpha)}
                style={{
                  padding: '1px 5px',
                  border: 'none',
                  borderRadius: '3px',
                  fontSize: '0.5625rem',
                  fontFamily: 'var(--font-mono)',
                  cursor: 'pointer',
                  background: emaAlpha === opt.alpha ? 'var(--accent-dim)' : 'transparent',
                  color: emaAlpha === opt.alpha ? 'var(--accent)' : 'var(--text-muted)',
                  fontWeight: emaAlpha === opt.alpha ? 600 : 400,
                }}
              >
                {opt.label}
              </button>
            ))}
          </div>
          {onStepClick && (
            <span style={{ fontSize: '0.5rem', color: 'var(--text-muted)', fontStyle: 'italic', marginLeft: '0.25rem' }}>
              click → rollouts
            </span>
          )}
        </div>
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart
          data={data}
          onClick={onStepClick ? (e: unknown) => {
            const ev = e as { activePayload?: { payload?: { step?: number } }[] };
            if (ev?.activePayload?.[0]?.payload?.step != null) {
              onStepClick(ev.activePayload[0].payload.step);
            }
          } : undefined}
          style={onStepClick ? { cursor: 'crosshair' } : undefined}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="step" stroke="var(--text-muted)" tick={{ fontSize: 10 }} />
          <YAxis stroke="var(--text-muted)" tick={{ fontSize: 10 }} width={55} />
          <Tooltip
            contentStyle={{
              background: 'var(--bg-surface)',
              border: '1px solid var(--border)',
              borderRadius: '6px',
              fontSize: '0.75rem',
              padding: '0.375rem 0.5rem',
            }}
            labelStyle={{ color: 'var(--text-primary)', fontWeight: 600, marginBottom: '0.25rem' }}
            itemStyle={{ padding: '1px 0' }}
          />
          {metricKeys.map((key, i) => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              stroke={COLORS[i % COLORS.length]}
              dot={false}
              strokeWidth={smoothedData ? 0.5 : 1.5}
              strokeOpacity={hidden.has(key) ? 0 : (smoothedData ? 0.3 : 1)}
              name={shortName(key)}
              connectNulls
              hide={hidden.has(key)}
            />
          ))}
          {/* Smoothed overlay lines */}
          {smoothedData && metricKeys.map((key, i) => (
            <Line
              key={`${key}_ema`}
              data={smoothedData}
              type="monotone"
              dataKey={key}
              stroke={COLORS[i % COLORS.length]}
              dot={false}
              strokeWidth={2}
              strokeOpacity={hidden.has(key) ? 0 : 1}
              name={`${shortName(key)} (EMA)`}
              connectNulls
              hide={hidden.has(key)}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      {/* Legend */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem 0.625rem', marginTop: '0.375rem', paddingLeft: '0.25rem' }}>
        {metricKeys.map((key, i) => (
          <button
            key={key}
            onClick={() => toggleKey(key)}
            style={{
              display: 'flex', alignItems: 'center', gap: '0.25rem',
              background: 'none', border: 'none', cursor: 'pointer',
              fontSize: '0.6875rem', fontFamily: 'var(--font-mono)',
              color: hidden.has(key) ? 'var(--text-muted)' : 'var(--text-secondary)',
              opacity: hidden.has(key) ? 0.5 : 1,
              padding: '0.125rem 0',
              textDecoration: hidden.has(key) ? 'line-through' : 'none',
            }}
          >
            <span style={{
              width: 8, height: 8, borderRadius: '50%',
              background: COLORS[i % COLORS.length],
              opacity: hidden.has(key) ? 0.3 : 1, flexShrink: 0,
            }} />
            {shortName(key)}
          </button>
        ))}
      </div>
    </div>
  );
}

export function MetricsPanel({ runId, onStepClick }: Props) {
  const [records, setRecords] = useState<MetricRecord[]>([]);
  const [keys, setKeys] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    Promise.all([api.getMetrics(runId), api.getMetricKeys(runId)])
      .then(([metricsResp, metricKeys]) => {
        setRecords(metricsResp.records);
        setKeys(metricKeys);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));

    function handleMessage(event: MessageEvent) {
      try {
        const record = JSON.parse(event.data) as MetricRecord;
        setRecords((prev) => [...prev, record]);
        const newKeys = Object.keys(record).filter((k) => k !== 'step');
        setKeys((prev) => {
          const existing = new Set(prev);
          const added = newKeys.filter((k) => !existing.has(k));
          return added.length > 0 ? [...prev, ...added] : prev;
        });
      } catch { /* ignore keepalives */ }
    }

    function connect() {
      const es = new EventSource(api.metricsStreamUrl(runId));
      esRef.current = es;
      es.onmessage = handleMessage;
      es.onerror = () => {
        es.close();
        if (esRef.current === es) {
          esRef.current = null;
          setTimeout(() => { if (esRef.current === null) connect(); }, 5000);
        }
      };
    }

    connect();
    return () => { if (esRef.current) { esRef.current.close(); esRef.current = null; } };
  }, [runId]);

  const groups = useMemo(() => groupMetricKeys(keys), [keys]);

  if (loading) return <div className="loading">Loading metrics...</div>;
  if (error) return <div className="empty-state">{error}</div>;
  if (records.length === 0) return <div className="empty-state">No metrics data yet</div>;

  return (
    <div className="charts-grid">
      {Array.from(groups.entries()).map(([prefix, groupKeys]) => (
        <MetricChart key={prefix} prefix={prefix} metricKeys={groupKeys} data={records} onStepClick={onStepClick} />
      ))}
    </div>
  );
}

import { useEffect, useState, useRef } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import { api } from '../api/client';
import type { MetricRecord } from '../api/types';

// Color palette for chart lines
const COLORS = [
  '#6366f1', '#22c55e', '#f59e0b', '#ef4444', '#06b6d4',
  '#ec4899', '#8b5cf6', '#14b8a6', '#f97316', '#64748b',
];

interface Props {
  runId: string;
}

/** Group metric keys by their prefix (e.g., "optim/", "env/", "time/"). */
function groupMetricKeys(keys: string[]): Map<string, string[]> {
  const groups = new Map<string, string[]>();
  for (const key of keys) {
    // Skip noisy aggregate keys
    if (key.endsWith(':total') || key.endsWith(':count')) continue;

    const slashIdx = key.indexOf('/');
    const prefix = slashIdx > 0 ? key.substring(0, slashIdx) : 'general';
    if (!groups.has(prefix)) groups.set(prefix, []);
    groups.get(prefix)!.push(key);
  }
  return groups;
}

export function MetricsPanel({ runId }: Props) {
  const [records, setRecords] = useState<MetricRecord[]>([]);
  const [keys, setKeys] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    Promise.all([
      api.getMetrics(runId),
      api.getMetricKeys(runId),
    ])
      .then(([metricsResp, metricKeys]) => {
        setRecords(metricsResp.records);
        setKeys(metricKeys);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));

    // Connect SSE for live updates
    const url = api.metricsStreamUrl(runId);
    const es = new EventSource(url);
    esRef.current = es;

    es.onmessage = (event) => {
      try {
        const record = JSON.parse(event.data) as MetricRecord;
        setRecords((prev) => [...prev, record]);
        // Update keys if new ones appear
        const newKeys = Object.keys(record).filter((k) => k !== 'step');
        setKeys((prev) => {
          const existing = new Set(prev);
          const added = newKeys.filter((k) => !existing.has(k));
          return added.length > 0 ? [...prev, ...added] : prev;
        });
      } catch {
        // Ignore parse errors
      }
    };

    return () => {
      es.close();
      esRef.current = null;
    };
  }, [runId]);

  if (loading) return <div className="loading">Loading metrics...</div>;
  if (error) return <div className="empty-state">{error}</div>;
  if (records.length === 0) return <div className="empty-state">No metrics data</div>;

  const groups = groupMetricKeys(keys);

  return (
    <div className="charts-grid">
      {Array.from(groups.entries()).map(([prefix, groupKeys]) => (
        <div key={prefix} className="chart-card">
          <div className="chart-title">{prefix}</div>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={records}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis
                dataKey="step"
                stroke="var(--text-muted)"
                tick={{ fontSize: 11 }}
              />
              <YAxis stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  background: 'var(--bg-secondary)',
                  border: '1px solid var(--border)',
                  borderRadius: '6px',
                  fontSize: '0.8rem',
                }}
                labelStyle={{ color: 'var(--text-primary)' }}
              />
              {groupKeys.map((key, i) => (
                <Line
                  key={key}
                  type="monotone"
                  dataKey={key}
                  stroke={COLORS[i % COLORS.length]}
                  dot={false}
                  strokeWidth={1.5}
                  name={key.includes('/') ? key.split('/').slice(1).join('/') : key}
                  connectNulls
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      ))}
    </div>
  );
}

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
  '#6366f1', '#22c55e', '#f59e0b', '#ef4444', '#06b6d4',
  '#ec4899', '#8b5cf6', '#14b8a6', '#f97316', '#64748b',
];

interface Props {
  runId: string;
}

function groupMetricKeys(keys: string[]): Map<string, string[]> {
  const groups = new Map<string, string[]>();
  for (const key of keys) {
    if (key.endsWith(':total') || key.endsWith(':count')) continue;
    const slashIdx = key.indexOf('/');
    const prefix = slashIdx > 0 ? key.substring(0, slashIdx) : 'general';
    if (!groups.has(prefix)) groups.set(prefix, []);
    groups.get(prefix)!.push(key);
  }
  return groups;
}

function MetricChart({ prefix, metricKeys, data }: { prefix: string; metricKeys: string[]; data: MetricRecord[] }) {
  const [hidden, setHidden] = useState<Set<string>>(new Set());

  const toggleKey = (key: string) => {
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const shortName = (key: string) => key.includes('/') ? key.split('/').slice(1).join('/') : key;

  return (
    <div className="chart-card">
      <div className="chart-title">{prefix}</div>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data}>
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
              strokeWidth={hidden.has(key) ? 0 : 1.5}
              strokeOpacity={hidden.has(key) ? 0 : 1}
              name={shortName(key)}
              connectNulls
              hide={hidden.has(key)}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      {/* Clickable legend */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem 0.625rem', marginTop: '0.375rem', paddingLeft: '0.25rem' }}>
        {metricKeys.map((key, i) => (
          <button
            key={key}
            onClick={() => toggleKey(key)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.25rem',
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              fontSize: '0.6875rem',
              color: hidden.has(key) ? 'var(--text-muted)' : 'var(--text-secondary)',
              opacity: hidden.has(key) ? 0.5 : 1,
              padding: '0.125rem 0',
              textDecoration: hidden.has(key) ? 'line-through' : 'none',
            }}
          >
            <span style={{
              width: 8, height: 8, borderRadius: '50%',
              background: COLORS[i % COLORS.length],
              opacity: hidden.has(key) ? 0.3 : 1,
              flexShrink: 0,
            }} />
            {shortName(key)}
          </button>
        ))}
      </div>
    </div>
  );
}

export function MetricsPanel({ runId }: Props) {
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
        // Only reconnect if this is still the active connection
        if (esRef.current === es) {
          esRef.current = null;
          setTimeout(() => {
            if (esRef.current === null) connect();
          }, 5000);
        }
      };
    }

    connect();

    return () => {
      if (esRef.current) {
        esRef.current.close();
        esRef.current = null;
      }
    };
  }, [runId]);

  const groups = useMemo(() => groupMetricKeys(keys), [keys]);

  if (loading) return <div className="loading">Loading metrics...</div>;
  if (error) return <div className="empty-state">{error}</div>;
  if (records.length === 0) return <div className="empty-state">No metrics data yet</div>;

  return (
    <div className="charts-grid">
      {Array.from(groups.entries()).map(([prefix, groupKeys]) => (
        <MetricChart key={prefix} prefix={prefix} metricKeys={groupKeys} data={records} />
      ))}
    </div>
  );
}

import { useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { api } from '../api/client';
import { MetricsPanel } from '../components/MetricsPanel';
import { RolloutBrowser } from '../components/RolloutBrowser';
import { TimingPanel } from '../components/TimingPanel';
import { StatusBadge, TypeBadge } from '../utils/shared';
import type { IterationInfo, MetricRecord, RunInfo } from '../api/types';

type Tab = 'metrics' | 'rollouts' | 'timing' | 'config';

/** Pick the latest value and trend direction for a metric key. */
function getStatFromMetrics(records: MetricRecord[], key: string): { value: number | null; trend: 'up' | 'down' | 'flat'; sparkData: number[] } {
  const values = records.map((r) => r[key]).filter((v): v is number => typeof v === 'number');
  if (values.length === 0) return { value: null, trend: 'flat', sparkData: [] };
  const latest = values[values.length - 1];
  const prev = values.length > 1 ? values[values.length - 2] : latest;
  const trend: 'up' | 'down' | 'flat' = latest > prev + 0.001 ? 'up' : latest < prev - 0.001 ? 'down' : 'flat';
  return { value: latest, trend, sparkData: values.slice(-30) };
}

function MiniSparkline({ data, color }: { data: number[]; color: string }) {
  if (data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const h = 28;
  const w = 80;
  const points = data.map((v, i) => `${(i / (data.length - 1)) * w},${h - ((v - min) / range) * (h - 4) - 2}`).join(' ');
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ display: 'block' }}>
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" />
    </svg>
  );
}

function formatValue(v: number | null, isPercent?: boolean): string {
  if (v === null) return '-';
  if (isPercent) return `${(v * 100).toFixed(1)}%`;
  if (Math.abs(v) >= 100) return v.toFixed(1);
  if (Math.abs(v) >= 1) return v.toFixed(2);
  return v.toFixed(4);
}

const TREND_ARROWS: Record<string, string> = { up: '\u2191', down: '\u2193', flat: '' };
const TREND_COLORS: Record<string, string> = { up: 'var(--accent)', down: 'var(--error)', flat: 'var(--text-muted)' };

export function RunDetailPage() {
  const { runId } = useParams<{ runId: string }>();
  const [run, setRun] = useState<RunInfo | null>(null);
  const [iterations, setIterations] = useState<IterationInfo[]>([]);
  const [metrics, setMetrics] = useState<MetricRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>('metrics');
  const [visitedTabs, setVisitedTabs] = useState<Set<Tab>>(new Set(['metrics']));
  const [jumpToStep, setJumpToStep] = useState<number | null>(null);

  useEffect(() => {
    if (!runId) return;
    Promise.all([
      api.getRun(runId),
      api.listIterations(runId),
      api.getMetrics(runId),
    ])
      .then(([runData, iters, metricsResp]) => {
        setRun(runData);
        setIterations(iters);
        setMetrics(metricsResp.records);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [runId]);

  const switchTab = (tab: Tab) => {
    setActiveTab(tab);
    setVisitedTabs((prev) => new Set(prev).add(tab));
  };

  const handleMetricStepClick = (step: number) => {
    setJumpToStep(step);
    switchTab('rollouts');
  };

  // Compute stat cards from latest metrics
  const reward = useMemo(() => getStatFromMetrics(metrics, 'env/all/reward/total'), [metrics]);
  const correct = useMemo(() => getStatFromMetrics(metrics, 'env/all/correct'), [metrics]);
  const kl = useMemo(() => getStatFromMetrics(metrics, 'optim/kl_sample_train_v1'), [metrics]);
  const speed = useMemo(() => getStatFromMetrics(metrics, 'time/total'), [metrics]);

  if (loading) return <div className="loading">Loading run...</div>;
  if (error) return <div className="error-msg">{error}</div>;
  if (!run || !runId) return <div className="error-msg">Run not found</div>;

  const hasIterations = iterations.some((it) => it.has_train_rollouts);
  const hasTiming = run.has_timing;

  const tabs: { id: Tab; label: string; disabled?: boolean }[] = [
    { id: 'metrics', label: 'Metrics' },
    { id: 'rollouts', label: 'Rollouts', disabled: !hasIterations },
    { id: 'timing', label: 'Timing', disabled: !hasTiming },
    { id: 'config', label: 'Config' },
  ];

  return (
    <div>
      <div className="breadcrumb">
        <Link to="/">Dashboard</Link>
        <span>/</span>
        <span>{runId}</span>
      </div>

      {/* Run header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '0.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <h2 className="page-title">{runId}</h2>
          <TypeBadge type={run.training_type} />
          <StatusBadge status={run.status} />
        </div>
        <div className="text-muted" style={{ fontSize: '0.8125rem' }}>
          {run.config_summary?.model_name != null && String(run.config_summary.model_name)}
          {run.total_steps != null && <span> · step {run.total_steps}</span>}
        </div>
      </div>

      {/* Stat cards — answer "is training working?" at a glance */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '0.5rem', marginBottom: '1rem' }}>
        <StatCard label="Reward" value={formatValue(reward.value)} trend={reward.trend} spark={reward.sparkData} color="var(--accent)" />
        <StatCard label="Correct" value={formatValue(correct.value, true)} trend={correct.trend} spark={correct.sparkData} color="var(--cyan)" />
        <StatCard label="KL Divergence" value={formatValue(kl.value)} trend={kl.trend} spark={kl.sparkData} color="var(--warning)" invertTrend />
        <StatCard label="Step Time" value={speed.value !== null ? `${speed.value.toFixed(1)}s` : '-'} trend={speed.trend} spark={speed.sparkData} color="var(--purple)" invertTrend />
      </div>

      {/* 4 tabs */}
      <div className="tabs">
        {tabs.map(({ id, label, disabled }) => (
          <button
            key={id}
            className={`tab ${activeTab === id ? 'active' : ''}`}
            onClick={() => !disabled && switchTab(id)}
            style={disabled ? { opacity: 0.3, cursor: 'default' } : undefined}
          >
            {label}
          </button>
        ))}
      </div>

      <div style={{ display: activeTab === 'metrics' ? 'block' : 'none' }}>
        {visitedTabs.has('metrics') && <MetricsPanel runId={runId} onStepClick={handleMetricStepClick} />}
      </div>
      <div style={{ display: activeTab === 'rollouts' ? 'block' : 'none' }}>
        {visitedTabs.has('rollouts') && <RolloutBrowser runId={runId} iterations={iterations} jumpToStep={jumpToStep} />}
      </div>
      <div style={{ display: activeTab === 'timing' ? 'block' : 'none' }}>
        {visitedTabs.has('timing') && <TimingPanel runId={runId} />}
      </div>
      <div style={{ display: activeTab === 'config' ? 'block' : 'none' }}>
        {visitedTabs.has('config') && <ConfigPanel config={run.config ?? {}} />}
      </div>
    </div>
  );
}

function StatCard({ label, value, trend, spark, color, invertTrend }: {
  label: string;
  value: string;
  trend: 'up' | 'down' | 'flat';
  spark: number[];
  color: string;
  invertTrend?: boolean;
}) {
  // For KL and speed, "up" is bad (red) and "down" is good (green)
  const trendColor = invertTrend
    ? (trend === 'up' ? 'var(--error)' : trend === 'down' ? 'var(--accent)' : 'var(--text-muted)')
    : TREND_COLORS[trend];

  return (
    <div className="card" style={{ padding: '0.625rem 0.75rem' }}>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.5625rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.25rem' }}>
        {label}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <span className="mono" style={{ fontSize: '1.25rem', fontWeight: 700, color }}>
            {value}
          </span>
          {trend !== 'flat' && (
            <span style={{ fontSize: '0.8125rem', marginLeft: '0.25rem', color: trendColor, fontWeight: 600 }}>
              {TREND_ARROWS[trend]}
            </span>
          )}
        </div>
        <MiniSparkline data={spark} color={color} />
      </div>
    </div>
  );
}

function ConfigPanel({ config }: { config: Record<string, unknown> }) {
  const [search, setSearch] = useState('');
  const configStr = JSON.stringify(config, null, 2);
  const lines = configStr.split('\n');
  const filtered = search
    ? lines.filter((line) => line.toLowerCase().includes(search.toLowerCase()))
    : lines;

  const topLevelKeys = Object.keys(config);
  const scalarKeys = topLevelKeys.filter((k) => typeof config[k] !== 'object' || config[k] === null);
  const objectKeys = topLevelKeys.filter((k) => typeof config[k] === 'object' && config[k] !== null);
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());

  return (
    <div>
      <div className="card" style={{ marginBottom: '0.75rem' }}>
        <div className="card-header">
          <span>Configuration</span>
          <input type="text" placeholder="Search..." value={search} onChange={(e) => setSearch(e.target.value)} style={{ width: '180px' }} />
        </div>
        {search ? (
          <pre className="mono" style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6, fontSize: '0.6875rem' }}>{filtered.join('\n')}</pre>
        ) : (
          <div style={{ fontSize: '0.8125rem' }}>
            <table style={{ marginBottom: '0.75rem' }}>
              <tbody>
                {scalarKeys.map((key) => (
                  <tr key={key} style={{ cursor: 'default' }}>
                    <td className="mono text-muted" style={{ width: '200px', fontSize: '0.75rem' }}>{key}</td>
                    <td className="mono" style={{ fontSize: '0.75rem' }}>{String(config[key] ?? 'null')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {objectKeys.map((key) => (
              <div key={key} style={{ marginBottom: '0.375rem' }}>
                <div
                  onClick={() => setExpandedSections((prev) => { const n = new Set(prev); if (n.has(key)) n.delete(key); else n.add(key); return n; })}
                  style={{ cursor: 'pointer', padding: '0.375rem 0.5rem', background: 'var(--bg-elevated)', borderRadius: '4px', fontFamily: 'var(--font-mono)', fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '0.375rem' }}
                >
                  <span>{expandedSections.has(key) ? '\u25bc' : '\u25b6'}</span>
                  <span>{key}</span>
                  <span className="text-muted" style={{ fontWeight: 400, fontSize: '0.625rem' }}>
                    {Array.isArray(config[key]) ? `[${(config[key] as unknown[]).length} items]` : `{${Object.keys(config[key] as object).length} fields}`}
                  </span>
                </div>
                {expandedSections.has(key) && (
                  <pre className="mono" style={{ whiteSpace: 'pre-wrap', lineHeight: 1.5, fontSize: '0.6875rem', padding: '0.5rem', margin: '0.25rem 0 0 1rem', background: 'var(--bg-elevated)', borderRadius: '4px', maxHeight: '300px', overflow: 'auto' }}>
                    {JSON.stringify(config[key], null, 2)}
                  </pre>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

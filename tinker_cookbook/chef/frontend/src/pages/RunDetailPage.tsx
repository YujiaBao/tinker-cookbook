import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { api } from '../api/client';
import { EvalSummaryPanel } from '../components/EvalSummaryPanel';
import { MetricsPanel } from '../components/MetricsPanel';
import { RolloutBrowser } from '../components/RolloutBrowser';
import { RunOverviewPanel } from '../components/RunOverviewPanel';
import { TimingPanel } from '../components/TimingPanel';
import { StatusBadge, TypeBadge } from '../utils/shared';
import type { IterationInfo, RunInfo } from '../api/types';

type Tab = 'overview' | 'metrics' | 'rollouts' | 'checkpoints' | 'timing' | 'config';

export function RunDetailPage() {
  const { runId } = useParams<{ runId: string }>();
  const [run, setRun] = useState<RunInfo | null>(null);
  const [iterations, setIterations] = useState<IterationInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>('overview');
  const [visitedTabs, setVisitedTabs] = useState<Set<Tab>>(new Set(['overview']));

  useEffect(() => {
    if (!runId) return;
    Promise.all([api.getRun(runId), api.listIterations(runId)])
      .then(([runData, iters]) => { setRun(runData); setIterations(iters); })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [runId]);

  const switchTab = (tab: Tab) => {
    setActiveTab(tab);
    setVisitedTabs((prev) => new Set(prev).add(tab));
  };

  if (loading) return <div className="loading">Loading run...</div>;
  if (error) return <div className="error-msg">{error}</div>;
  if (!run || !runId) return <div className="error-msg">Run not found</div>;

  const hasIterations = iterations.some((it) => it.has_train_rollouts);
  const hasTiming = run.has_timing;

  const tabs: { id: Tab; label: string; disabled?: boolean }[] = [
    { id: 'overview', label: 'Overview' },
    { id: 'metrics', label: 'Metrics' },
    { id: 'rollouts', label: 'Rollouts', disabled: !hasIterations },
    { id: 'checkpoints', label: 'Checkpoints & Evals' },
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

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '0.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <h2 className="page-title">{runId}</h2>
          <TypeBadge type={run.training_type} />
          <StatusBadge status={run.status} />
        </div>
        <div className="text-muted" style={{ fontSize: '0.8125rem' }}>
          {run.config_summary?.model_name as string ?? ''}
          {run.total_steps != null && <span> · {run.total_steps} steps</span>}
        </div>
      </div>

      <div className="tabs">
        {tabs.map(({ id, label, disabled }) => (
          <button
            key={id}
            className={`tab ${activeTab === id ? 'active' : ''}`}
            onClick={() => !disabled && switchTab(id)}
            style={disabled ? { opacity: 0.4, cursor: 'default' } : undefined}
            title={disabled ? `No ${label.toLowerCase()} data` : undefined}
          >
            {label}
          </button>
        ))}
      </div>

      <div style={{ display: activeTab === 'overview' ? 'block' : 'none' }}>
        {visitedTabs.has('overview') && <RunOverviewPanel runId={runId} run={run} />}
      </div>
      <div style={{ display: activeTab === 'metrics' ? 'block' : 'none' }}>
        {visitedTabs.has('metrics') && <MetricsPanel runId={runId} />}
      </div>
      <div style={{ display: activeTab === 'rollouts' ? 'block' : 'none' }}>
        {visitedTabs.has('rollouts') && <RolloutBrowser runId={runId} iterations={iterations} />}
      </div>
      <div style={{ display: activeTab === 'checkpoints' ? 'block' : 'none' }}>
        {visitedTabs.has('checkpoints') && <EvalSummaryPanel runId={runId} />}
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

function ConfigPanel({ config }: { config: Record<string, unknown> }) {
  const [search, setSearch] = useState('');
  const configStr = JSON.stringify(config, null, 2);
  const lines = configStr.split('\n');
  const filtered = search
    ? lines.filter((line) => line.toLowerCase().includes(search.toLowerCase()))
    : lines;

  return (
    <div className="card">
      <div className="card-header">
        <span>Configuration</span>
        <input
          type="text"
          placeholder="Search config..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{
            padding: '0.25rem 0.5rem', borderRadius: '4px',
            border: '1px solid var(--border)', background: 'var(--bg-tertiary)',
            color: 'var(--text-primary)', fontSize: '0.75rem', width: '180px',
          }}
        />
      </div>
      <pre className="mono" style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6, fontSize: '0.75rem' }}>
        {filtered.join('\n')}
      </pre>
    </div>
  );
}

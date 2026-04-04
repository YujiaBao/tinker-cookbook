import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { api } from '../api/client';
import { CheckpointsPanel } from '../components/CheckpointsPanel';
import { EvalSummaryPanel } from '../components/EvalSummaryPanel';
import { MetricsPanel } from '../components/MetricsPanel';
import { RolloutBrowser } from '../components/RolloutBrowser';
import { RunOverviewPanel } from '../components/RunOverviewPanel';
import { TimingPanel } from '../components/TimingPanel';
import { StatusBadge, TypeBadge } from '../utils/shared';
import type { IterationInfo, RunInfo } from '../api/types';

type Tab = 'overview' | 'metrics' | 'rollouts' | 'checkpoints' | 'evals' | 'timing' | 'config';

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
    { id: 'checkpoints', label: 'Checkpoints', disabled: !run.has_checkpoints },
    { id: 'evals', label: 'Evals' },
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
            style={disabled ? { opacity: 0.3, cursor: 'default' } : undefined}
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
        {visitedTabs.has('checkpoints') && <CheckpointsPanel runId={runId} />}
      </div>
      <div style={{ display: activeTab === 'evals' ? 'block' : 'none' }}>
        {visitedTabs.has('evals') && <EvalSummaryPanel runId={runId} />}
      </div>
      <div style={{ display: activeTab === 'timing' ? 'block' : 'none' }}>
        {visitedTabs.has('timing') && <TimingPanel runId={runId} />}
      </div>
      <div style={{ display: activeTab === 'config' ? 'block' : 'none' }}>
        {visitedTabs.has('config') && <ConfigPanel runId={runId} config={run.config ?? {}} />}
      </div>
    </div>
  );
}

function ConfigPanel({ config }: { runId: string; config: Record<string, unknown> }) {
  const [search, setSearch] = useState('');
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());

  // Extract key info for the summary header
  const modelName = config.model_name as string ?? null;
  const loraRank = config.lora_rank as number ?? null;
  const lr = config.learning_rate as number ?? null;
  const lossFn = config.loss_fn as string ?? null;
  const datasetBuilder = config.dataset_builder;
  const ttl = config.ttl_seconds as number ?? null;

  const configStr = JSON.stringify(config, null, 2);
  const lines = configStr.split('\n');
  const filtered = search
    ? lines.filter((line) => line.toLowerCase().includes(search.toLowerCase()))
    : lines;

  // Group top-level keys
  const topLevelKeys = Object.keys(config);
  const scalarKeys = topLevelKeys.filter((k) => typeof config[k] !== 'object' || config[k] === null);
  const objectKeys = topLevelKeys.filter((k) => typeof config[k] === 'object' && config[k] !== null);

  const toggleSection = (key: string) => {
    setExpandedSections((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  return (
    <div>
      {/* Quick summary */}
      <div className="card" style={{ marginBottom: '0.75rem' }}>
        <div className="card-header">Training Configuration</div>
        <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap', fontSize: '0.8125rem' }}>
          {modelName && <div><span className="text-muted">Model: </span><span className="mono" style={{ fontWeight: 600 }}>{String(modelName)}</span></div>}
          {lr != null && <div><span className="text-muted">LR: </span><span className="mono">{String(lr)}</span></div>}
          {loraRank != null && <div><span className="text-muted">LoRA rank: </span><span className="mono">{String(loraRank)}</span></div>}
          {lossFn && <div><span className="text-muted">Loss: </span><span className="mono">{String(lossFn)}</span></div>}
          {ttl != null && <div><span className="text-muted">TTL: </span><span className="mono">{Math.round(Number(ttl) / 3600)}h</span></div>}
          {typeof datasetBuilder === 'object' && datasetBuilder !== null && (
            <div><span className="text-muted">Dataset: </span><span className="mono">{String((datasetBuilder as Record<string, unknown>).__type__ ?? 'custom')}</span></div>
          )}
        </div>
      </div>

      {/* Structured config viewer */}
      <div className="card">
        <div className="card-header">
          <span>Full Configuration</span>
          <input
            type="text"
            placeholder="Search..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ width: '180px' }}
          />
        </div>

        {search ? (
          <pre className="mono" style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6, fontSize: '0.6875rem' }}>
            {filtered.join('\n')}
          </pre>
        ) : (
          <div style={{ fontSize: '0.8125rem' }}>
            {/* Scalar fields */}
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

            {/* Nested objects (collapsible) */}
            {objectKeys.map((key) => (
              <div key={key} style={{ marginBottom: '0.375rem' }}>
                <div
                  onClick={() => toggleSection(key)}
                  style={{
                    cursor: 'pointer',
                    padding: '0.375rem 0.5rem',
                    background: 'var(--bg-elevated)',
                    borderRadius: '4px',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.75rem',
                    fontWeight: 600,
                    color: 'var(--text-secondary)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.375rem',
                  }}
                >
                  <span>{expandedSections.has(key) ? '\u25bc' : '\u25b6'}</span>
                  <span>{key}</span>
                  <span className="text-muted" style={{ fontWeight: 400, fontSize: '0.625rem' }}>
                    {Array.isArray(config[key])
                      ? `[${(config[key] as unknown[]).length} items]`
                      : `{${Object.keys(config[key] as object).length} fields}`}
                  </span>
                </div>
                {expandedSections.has(key) && (
                  <pre className="mono" style={{
                    whiteSpace: 'pre-wrap', lineHeight: 1.5, fontSize: '0.6875rem',
                    padding: '0.5rem', margin: '0.25rem 0 0 1rem',
                    background: 'var(--bg-elevated)', borderRadius: '4px',
                    maxHeight: '300px', overflow: 'auto',
                  }}>
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

import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { api } from '../api/client';
import { MetricsPanel } from '../components/MetricsPanel';
import { RolloutBrowser } from '../components/RolloutBrowser';
import { TimingPanel } from '../components/TimingPanel';
import type { IterationInfo, RunInfo } from '../api/types';

type Tab = 'metrics' | 'rollouts' | 'timing' | 'config';

export function RunDetailPage() {
  const { runId } = useParams<{ runId: string }>();
  const [run, setRun] = useState<RunInfo | null>(null);
  const [iterations, setIterations] = useState<IterationInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>('metrics');

  useEffect(() => {
    if (!runId) return;
    Promise.all([
      api.getRun(runId),
      api.listIterations(runId),
    ])
      .then(([runData, iters]) => {
        setRun(runData);
        setIterations(iters);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [runId]);

  if (loading) return <div className="loading">Loading run...</div>;
  if (error) return <div className="error-msg">{error}</div>;
  if (!run || !runId) return <div className="error-msg">Run not found</div>;

  return (
    <div>
      <div className="breadcrumb">
        <Link to="/">Runs</Link>
        <span>/</span>
        <span>{runId}</span>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <div>
          <h2>{runId}</h2>
          {run.config_summary && (
            <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginTop: '0.25rem' }}>
              {Object.entries(run.config_summary).map(([k, v]) => (
                <span key={k} className="tag" style={{ marginRight: '0.5rem' }}>
                  {k}: {String(v)}
                </span>
              ))}
            </div>
          )}
        </div>
        <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
          {run.total_steps && <span>{run.total_steps} steps</span>}
          {run.iteration_count > 0 && <span> · {run.iteration_count} iterations</span>}
        </div>
      </div>

      <div className="tabs">
        {(['metrics', 'rollouts', 'timing', 'config'] as Tab[]).map((tab) => (
          <button
            key={tab}
            className={`tab ${activeTab === tab ? 'active' : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {activeTab === 'metrics' && <MetricsPanel runId={runId} />}
      {activeTab === 'rollouts' && <RolloutBrowser runId={runId} iterations={iterations} />}
      {activeTab === 'timing' && <TimingPanel runId={runId} />}
      {activeTab === 'config' && <ConfigPanel config={run.config ?? {}} />}
    </div>
  );
}

function ConfigPanel({ config }: { config: Record<string, unknown> }) {
  return (
    <div className="card">
      <div className="card-header">Configuration</div>
      <pre className="mono" style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
        {JSON.stringify(config, null, 2)}
      </pre>
    </div>
  );
}

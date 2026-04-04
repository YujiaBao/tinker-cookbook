import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import type { RunInfo, ScoresTableRow } from '../api/types';

const TYPE_LABELS: Record<string, string> = { rl: 'RL', sl: 'SFT', dpo: 'DPO' };
const TYPE_COLORS: Record<string, string> = { rl: '#6366f1', sl: '#22c55e', dpo: '#f59e0b' };
const STATUS_LABELS: Record<string, string> = { running: 'Running', completed: 'Completed', idle: 'Idle' };

function scoreColor(score: number): string {
  if (score >= 0.8) return 'var(--success)';
  if (score >= 0.5) return 'var(--warning)';
  return 'var(--error)';
}

function timeAgo(ts: number): string {
  const seconds = Math.floor(Date.now() / 1000 - ts);
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

export function DashboardPage() {
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [scores, setScores] = useState<ScoresTableRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAllRuns, setShowAllRuns] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    Promise.all([
      api.listRuns(),
      api.getScoresTable().catch(() => [] as ScoresTableRow[]),
    ])
      .then(([runList, scoreList]) => {
        setRuns(runList);
        setScores(scoreList);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="loading">Loading dashboard...</div>;

  const activeRuns = runs.filter((r) => r.status === 'running');
  const recentRuns = runs.slice(0, 8);

  // Collect benchmark names from scores
  const benchmarks = new Set<string>();
  for (const row of scores) {
    for (const k of Object.keys(row.scores)) benchmarks.add(k);
  }

  return (
    <div>
      <h2 className="page-title">Dashboard</h2>
      <div className="page-subtitle">
        {activeRuns.length > 0 && <span style={{ color: 'var(--success)' }}>{activeRuns.length} running</span>}
        {activeRuns.length > 0 && runs.length > activeRuns.length && ' · '}
        {runs.length - activeRuns.length > 0 && `${runs.length - activeRuns.length} completed`}
        {runs.length === 0 && 'No training runs found'}
      </div>

      {/* Active Runs */}
      {activeRuns.length > 0 && (
        <div style={{ marginBottom: '1.25rem' }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.5rem' }}>
            Active Runs
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '0.625rem' }}>
            {activeRuns.map((run) => (
              <RunCard key={run.run_id} run={run} onClick={() => navigate(`/runs/${run.run_id}`)} />
            ))}
          </div>
        </div>
      )}

      {/* Latest Eval Scores */}
      {scores.length > 0 && (
        <div className="card" style={{ marginBottom: '1.25rem', overflow: 'auto' }}>
          <div className="card-header">
            <span className="card-title">Eval Progression</span>
          </div>
          <table>
            <thead>
              <tr>
                <th>Checkpoint</th>
                <th>Model</th>
                {[...benchmarks].sort().map((b) => <th key={b}>{b}</th>)}
              </tr>
            </thead>
            <tbody>
              {scores.slice(-5).map((row, i, arr) => {
                const prev = i > 0 ? arr[i - 1] : undefined;
                return (
                  <tr key={row.run_id} onClick={() => navigate(`/eval/${row.run_id}`)} style={{ cursor: 'pointer' }}>
                    <td className="mono" style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                      {row.checkpoint_name ?? row.run_id}
                    </td>
                    <td>{row.model_name}</td>
                    {[...benchmarks].sort().map((b) => {
                      const score = row.scores[b];
                      const delta = score !== undefined && prev?.scores[b] !== undefined
                        ? score - prev.scores[b] : undefined;
                      return (
                        <td key={b} className="mono">
                          {score !== undefined ? (
                            <>
                              <span style={{ color: scoreColor(score) }}>
                                {(score * 100).toFixed(1)}%
                              </span>
                              {delta !== undefined && delta !== 0 && (
                                <span style={{ fontSize: '0.5625rem', marginLeft: '0.1875rem', color: delta > 0 ? 'var(--success)' : 'var(--error)' }}>
                                  {delta > 0 ? '+' : ''}{(delta * 100).toFixed(1)}
                                </span>
                              )}
                            </>
                          ) : <span className="text-muted">-</span>}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* All Runs */}
      <div className="card" style={{ padding: 0, overflow: 'auto' }}>
        <div className="card-header" style={{ padding: '0.75rem 1rem', cursor: 'pointer' }} onClick={() => setShowAllRuns(!showAllRuns)}>
          <span className="card-title">All Runs ({runs.length})</span>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            {showAllRuns || runs.length <= 5 ? '' : 'Click to expand'}
          </span>
        </div>
        <table>
          <thead>
            <tr>
              <th>Run</th>
              <th>Type</th>
              <th>Model</th>
              <th>Status</th>
              <th>Steps</th>
              <th>Updated</th>
            </tr>
          </thead>
          <tbody>
            {(showAllRuns || runs.length <= 5 ? recentRuns : recentRuns.slice(0, 3)).map((run) => (
              <tr key={run.run_id} onClick={() => navigate(`/runs/${run.run_id}`)}>
                <td className="mono" style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{run.run_id}</td>
                <td>
                  {run.training_type && (
                    <span className="tag" style={{ background: `${TYPE_COLORS[run.training_type]}22`, color: TYPE_COLORS[run.training_type] }}>
                      {TYPE_LABELS[run.training_type]}
                    </span>
                  )}
                </td>
                <td>{run.config_summary?.model_name as string ?? '-'}</td>
                <td>
                  <StatusBadge status={run.status} />
                </td>
                <td className="mono">{run.latest_step ?? '-'}</td>
                <td className="text-muted" style={{ fontSize: '0.75rem' }}>
                  {run.last_updated ? timeAgo(run.last_updated) : '-'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {runs.length === 0 && (
        <div className="empty-state" style={{ marginTop: '2rem' }}>
          <p>Point <code className="mono">tinker-chef serve</code> at a directory containing training run outputs.</p>
        </div>
      )}
    </div>
  );
}

function RunCard({ run, onClick }: { run: RunInfo; onClick: () => void }) {
  return (
    <div className="card" onClick={onClick} style={{ cursor: 'pointer', padding: '0.875rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.375rem' }}>
        <div>
          <div style={{ fontWeight: 600, fontSize: '0.9375rem' }}>{run.run_id}</div>
          <div className="text-muted" style={{ fontSize: '0.75rem' }}>
            {run.config_summary?.model_name as string ?? 'Unknown model'}
          </div>
        </div>
        <div style={{ display: 'flex', gap: '0.375rem', alignItems: 'center' }}>
          {run.training_type && (
            <span className="tag" style={{ background: `${TYPE_COLORS[run.training_type]}22`, color: TYPE_COLORS[run.training_type] }}>
              {TYPE_LABELS[run.training_type]}
            </span>
          )}
          <StatusBadge status={run.status} />
        </div>
      </div>
      <div style={{ display: 'flex', gap: '1rem', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
        {run.latest_step != null && <span>Step {run.latest_step}</span>}
        {run.iteration_count > 0 && <span>{run.iteration_count} iterations</span>}
        {run.last_updated && <span>{timeAgo(run.last_updated)}</span>}
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, { bg: string; fg: string }> = {
    running: { bg: 'rgba(34, 197, 94, 0.15)', fg: 'var(--success)' },
    completed: { bg: 'rgba(99, 102, 241, 0.15)', fg: '#818cf8' },
    idle: { bg: 'rgba(100, 116, 139, 0.15)', fg: 'var(--text-muted)' },
  };
  const c = colors[status] ?? colors.idle;
  return (
    <span className="badge" style={{ background: c.bg, color: c.fg }}>
      {status === 'running' && <span style={{ width: 6, height: 6, borderRadius: '50%', background: c.fg, animation: 'pulse 2s infinite' }} />}
      {STATUS_LABELS[status] ?? status}
    </span>
  );
}

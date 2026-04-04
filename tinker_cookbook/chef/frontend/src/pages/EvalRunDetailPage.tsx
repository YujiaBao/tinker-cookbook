import { useEffect, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { api } from '../api/client';
import type { EvalRunDetail, EvalTrajectorySummary } from '../api/types';

export function EvalRunDetailPage() {
  const { evalRunId } = useParams<{ evalRunId: string }>();
  const navigate = useNavigate();
  const [run, setRun] = useState<EvalRunDetail | null>(null);
  const [selectedBenchmark, setSelectedBenchmark] = useState<string | null>(null);
  const [trajectories, setTrajectories] = useState<EvalTrajectorySummary[]>([]);
  const [filter, setFilter] = useState<'all' | 'correct' | 'errors'>('all');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!evalRunId) return;
    api
      .getEvalRun(evalRunId)
      .then((data) => {
        setRun(data);
        if (data.benchmarks.length > 0) {
          setSelectedBenchmark(data.benchmarks[0]);
        }
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [evalRunId]);

  // Load trajectories when benchmark or filter changes
  useEffect(() => {
    if (!evalRunId || !selectedBenchmark) return;
    api
      .getEvalTrajectories(evalRunId, selectedBenchmark, {
        correct_only: filter === 'correct',
        errors_only: filter === 'errors',
      })
      .then((resp) => setTrajectories(resp.trajectories))
      .catch(() => setTrajectories([]));
  }, [evalRunId, selectedBenchmark, filter]);

  if (loading) return <div className="loading">Loading eval run...</div>;
  if (!run || !evalRunId) return <div className="empty-state">Eval run not found</div>;

  return (
    <div>
      <div className="breadcrumb">
        <Link to="/eval">Eval</Link>
        <span>/</span>
        <span>{evalRunId}</span>
      </div>

      <h2 style={{ marginBottom: '0.5rem' }}>{evalRunId}</h2>
      <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginBottom: '1rem' }}>
        {(run.metadata as Record<string, unknown>).model_name as string} ·{' '}
        {(run.metadata as Record<string, unknown>).checkpoint_name as string ?? 'No checkpoint'}
      </div>

      {/* Benchmark results cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '12px', marginBottom: '1.5rem' }}>
        {Object.entries(run.results).map(([name, result]) => (
          <div
            key={name}
            className="card"
            style={{
              cursor: 'pointer',
              borderColor: selectedBenchmark === name ? 'var(--accent)' : undefined,
            }}
            onClick={() => setSelectedBenchmark(name)}
          >
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '4px' }}>
              {name}
            </div>
            <div style={{ fontSize: '1.5rem', fontWeight: 700, color: scoreColor(result.score) }}>
              {(result.score * 100).toFixed(1)}%
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
              {result.num_correct}/{result.num_examples}
              {result.num_errors > 0 && (
                <span style={{ color: 'var(--error)' }}> ({result.num_errors} errors)</span>
              )}
            </div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
              {result.time_seconds.toFixed(1)}s
            </div>
          </div>
        ))}
      </div>

      {/* Trajectory list */}
      {selectedBenchmark && (
        <>
          <div className="filters-bar">
            <div className="filter-group">
              <span className="filter-label">Filter</span>
              <select value={filter} onChange={(e) => setFilter(e.target.value as typeof filter)}>
                <option value="all">All</option>
                <option value="correct">Correct only</option>
                <option value="errors">Errors only</option>
              </select>
            </div>
            <div style={{ marginLeft: 'auto', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              {trajectories.length} trajectories
            </div>
          </div>

          <div className="card" style={{ padding: 0, overflow: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Example ID</th>
                  <th>Reward</th>
                  <th>Turns</th>
                  <th>Time</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {trajectories.map((traj) => (
                  <tr
                    key={traj.idx}
                    onClick={() =>
                      navigate(`/eval/${evalRunId}/${selectedBenchmark}/${traj.idx}`)
                    }
                  >
                    <td className="mono">{traj.idx}</td>
                    <td className="mono" style={{ fontSize: '0.75rem' }}>
                      {traj.example_id ? traj.example_id.slice(0, 12) : '-'}
                    </td>
                    <td>
                      <span style={{ color: traj.reward > 0 ? 'var(--success)' : 'var(--error)', fontWeight: 600 }}>
                        {traj.reward.toFixed(1)}
                      </span>
                    </td>
                    <td className="mono">{traj.num_turns}</td>
                    <td className="mono">{traj.time_seconds.toFixed(1)}s</td>
                    <td>
                      {traj.error ? (
                        <span className="tag" style={{ background: 'rgba(239,68,68,0.15)', color: 'var(--error)' }}>
                          Error
                        </span>
                      ) : traj.reward > 0 ? (
                        <span className="tag" style={{ background: 'rgba(34,197,94,0.15)', color: 'var(--success)' }}>
                          Correct
                        </span>
                      ) : (
                        <span className="tag">Wrong</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

function scoreColor(score: number): string {
  if (score >= 0.8) return 'var(--success)';
  if (score >= 0.5) return 'var(--warning)';
  return 'var(--error)';
}

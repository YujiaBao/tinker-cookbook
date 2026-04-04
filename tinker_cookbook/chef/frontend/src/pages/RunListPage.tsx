import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import type { RunInfo } from '../api/types';

export function RunListPage() {
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    api.listRuns()
      .then(setRuns)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="loading">Discovering training runs...</div>;
  if (error) return <div className="error-msg">{error}</div>;
  if (runs.length === 0) {
    return (
      <div className="empty-state">
        <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>No training runs found</div>
        <p>Point <code className="mono">tinker-chef serve</code> at a directory containing <code className="mono">metrics.jsonl</code> files.</p>
      </div>
    );
  }

  const filtered = search
    ? runs.filter((r) =>
        r.run_id.toLowerCase().includes(search.toLowerCase()) ||
        String(r.config_summary?.model_name ?? '').toLowerCase().includes(search.toLowerCase())
      )
    : runs;

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <div>
          <h2 className="page-title">Training Runs</h2>
          <div className="page-subtitle">{runs.length} run{runs.length !== 1 ? 's' : ''} discovered</div>
        </div>
        <input
          type="text"
          placeholder="Search runs..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{
            padding: '0.375rem 0.75rem',
            borderRadius: '6px',
            border: '1px solid var(--border)',
            background: 'var(--bg-secondary)',
            color: 'var(--text-primary)',
            fontSize: '0.8125rem',
            width: '220px',
          }}
        />
      </div>
      <div className="card" style={{ padding: 0, overflow: 'auto' }}>
        <table>
          <thead>
            <tr>
              <th>Run</th>
              <th>Model</th>
              <th>Steps</th>
              <th>Iterations</th>
              <th>Available Data</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((run) => (
              <tr key={run.run_id} onClick={() => navigate(`/runs/${run.run_id}`)}>
                <td className="mono" style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{run.run_id}</td>
                <td>{run.config_summary?.model_name as string ?? <span className="text-muted">-</span>}</td>
                <td className="mono">{run.latest_step ?? '-'}</td>
                <td className="mono">{run.iteration_count}</td>
                <td>
                  {run.has_metrics && <span className="tag">metrics</span>}
                  {run.has_timing && <span className="tag">timing</span>}
                  {run.has_checkpoints && <span className="tag">ckpts</span>}
                  {run.iteration_count > 0 && <span className="tag">rollouts</span>}
                </td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr style={{ cursor: 'default' }}>
                <td colSpan={5} style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '2rem' }}>
                  No runs matching "{search}"
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

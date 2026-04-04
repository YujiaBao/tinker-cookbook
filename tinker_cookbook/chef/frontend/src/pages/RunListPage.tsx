import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import type { RunInfo } from '../api/types';

export function RunListPage() {
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    api.listRuns()
      .then(setRuns)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="loading">Loading runs...</div>;
  if (error) return <div className="error-msg">{error}</div>;
  if (runs.length === 0) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
        <h2 style={{ marginBottom: '0.5rem' }}>No training runs found</h2>
        <p style={{ color: 'var(--text-muted)' }}>
          Point tinker-chef at a directory containing metrics.jsonl files.
        </p>
      </div>
    );
  }

  return (
    <div>
      <h2 style={{ marginBottom: '1rem' }}>Training Runs</h2>
      <div className="card">
        <table>
          <thead>
            <tr>
              <th>Run ID</th>
              <th>Model</th>
              <th>Steps</th>
              <th>Iterations</th>
              <th>Data</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <tr key={run.run_id} onClick={() => navigate(`/runs/${run.run_id}`)}>
                <td className="mono">{run.run_id}</td>
                <td>{run.config_summary?.model_name as string ?? '—'}</td>
                <td className="mono">{run.latest_step ?? '—'}</td>
                <td className="mono">{run.iteration_count}</td>
                <td>
                  {run.has_metrics && <span className="tag">metrics</span>}
                  {run.has_timing && <span className="tag">timing</span>}
                  {run.has_checkpoints && <span className="tag">checkpoints</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

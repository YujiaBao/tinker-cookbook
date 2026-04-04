import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import type { EvalRunSummary, ScoresTableRow } from '../api/types';

export function EvalPage() {
  const [evalRuns, setEvalRuns] = useState<EvalRunSummary[]>([]);
  const [scoresTable, setScoresTable] = useState<ScoresTableRow[]>([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    Promise.all([api.listEvalRuns(), api.getScoresTable()])
      .then(([runs, scores]) => {
        setEvalRuns(runs);
        setScoresTable(scores);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="loading">Loading eval data...</div>;

  // Collect all unique benchmark names
  const allBenchmarks = new Set<string>();
  for (const row of scoresTable) {
    for (const key of Object.keys(row.scores)) {
      allBenchmarks.add(key);
    }
  }
  const benchmarks = [...allBenchmarks].sort();

  if (evalRuns.length === 0) {
    return (
      <div className="empty-state">
        <h2 style={{ marginBottom: '0.5rem', color: 'var(--text-primary)' }}>No Eval Data</h2>
        <p>No evaluation benchmark data was found.</p>
        <p style={{ marginTop: '0.5rem', fontSize: '0.8rem' }}>
          Run benchmarks with the eval framework and point tinker-chef at the eval store directory.
        </p>
      </div>
    );
  }

  return (
    <div>
      <h2 style={{ marginBottom: '0.5rem' }}>Evaluation Results</h2>
      <p className="text-muted" style={{ fontSize: '0.875rem', marginBottom: '1rem' }}>
        {evalRuns.length} eval run{evalRuns.length !== 1 ? 's' : ''} across {benchmarks.length} benchmark{benchmarks.length !== 1 ? 's' : ''}
      </p>

      {/* Scores matrix table */}
      {scoresTable.length > 0 && benchmarks.length > 0 && (
        <div className="card" style={{ marginBottom: '1.5rem', overflow: 'auto' }}>
          <div className="card-header">Scores Matrix</div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Checkpoint</th>
                <th>Model</th>
                {benchmarks.map((b) => (
                  <th key={b}>{b}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {scoresTable.map((row) => (
                <tr key={row.run_id} style={{ cursor: 'default' }}>
                  <td className="mono">{row.checkpoint_name ?? row.run_id}</td>
                  <td>{row.model_name}</td>
                  {benchmarks.map((b) => (
                    <td key={b} className="mono">
                      {row.scores[b] !== undefined ? (
                        <span style={{ color: scoreColor(row.scores[b]) }}>
                          {(row.scores[b] * 100).toFixed(1)}%
                        </span>
                      ) : (
                        <span className="text-muted">-</span>
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Eval run list */}
      <div className="card">
        <div className="card-header">Eval Runs</div>
        <table className="data-table">
          <thead>
            <tr>
              <th>Run ID</th>
              <th>Model</th>
              <th>Checkpoint</th>
              <th>Benchmarks</th>
              <th>Timestamp</th>
            </tr>
          </thead>
          <tbody>
            {evalRuns.map((run) => (
              <tr
                key={run.eval_run_id}
                onClick={() => navigate(`/eval/${run.eval_run_id}`)}
              >
                <td className="mono">{run.eval_run_id}</td>
                <td>{run.model_name}</td>
                <td className="mono">{run.checkpoint_name ?? '-'}</td>
                <td>
                  {run.benchmarks.map((b) => (
                    <span key={b} className="tag">{b}</span>
                  ))}
                </td>
                <td className="text-muted">{run.timestamp ?? '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function scoreColor(score: number): string {
  if (score >= 0.8) return 'var(--success)';
  if (score >= 0.5) return 'var(--warning)';
  return 'var(--error)';
}

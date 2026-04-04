import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import { scoreColor } from '../utils/shared';
import type { EvalRunSummary, ScoresTableRow } from '../api/types';

interface Props {
  runId: string;
}

export function EvalSummaryPanel({ runId }: Props) {
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
  }, [runId]);

  if (loading) return <div className="loading">Loading eval data...</div>;

  if (evalRuns.length === 0) {
    return (
      <div className="empty-state">
        <p>No evaluation data found.</p>
        <p style={{ marginTop: '0.5rem', fontSize: '0.75rem' }}>
          Run benchmarks with the eval framework to see results here.
        </p>
      </div>
    );
  }

  // Collect benchmarks
  const benchmarks = new Set<string>();
  for (const row of scoresTable) {
    for (const key of Object.keys(row.scores)) benchmarks.add(key);
  }
  const benchmarkList = [...benchmarks].sort();

  return (
    <div>
      {/* Score progression */}
      {scoresTable.length > 0 && benchmarkList.length > 0 && (
        <div className="card" style={{ marginBottom: '1rem', overflow: 'auto' }}>
          <div className="card-header">
            <span className="card-title">Benchmark Scores by Checkpoint</span>
          </div>
          <table>
            <thead>
              <tr>
                <th>Checkpoint</th>
                {benchmarkList.map((b) => <th key={b}>{b}</th>)}
              </tr>
            </thead>
            <tbody>
              {scoresTable.map((row, rowIdx) => {
                const prevRow = rowIdx > 0 ? scoresTable[rowIdx - 1] : undefined;
                return (
                  <tr
                    key={row.run_id}
                    onClick={() => navigate(`/eval/${row.run_id}`)}
                    style={{ cursor: 'pointer' }}
                  >
                    <td className="mono" style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                      {row.checkpoint_name ?? row.run_id}
                    </td>
                    {benchmarkList.map((b) => {
                      const score = row.scores[b];
                      const prev = prevRow?.scores[b];
                      const delta = score !== undefined && prev !== undefined ? score - prev : undefined;
                      return (
                        <td key={b} className="mono">
                          {score !== undefined ? (
                            <>
                              <span style={{ color: scoreColor(score) }}>
                                {(score * 100).toFixed(1)}%
                              </span>
                              {delta !== undefined && delta !== 0 && (
                                <span style={{
                                  fontSize: '0.5625rem',
                                  marginLeft: '0.1875rem',
                                  color: delta > 0 ? 'var(--success)' : 'var(--error)',
                                }}>
                                  {delta > 0 ? '+' : ''}{(delta * 100).toFixed(1)}
                                </span>
                              )}
                            </>
                          ) : (
                            <span className="text-muted">-</span>
                          )}
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

      {/* Recent eval runs */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">Eval Runs</span>
          <span style={{ fontSize: '0.6875rem', color: 'var(--text-muted)' }}>
            {evalRuns.length} run{evalRuns.length !== 1 ? 's' : ''}
          </span>
        </div>
        <table>
          <thead>
            <tr>
              <th>Run</th>
              <th>Checkpoint</th>
              <th>Benchmarks</th>
            </tr>
          </thead>
          <tbody>
            {evalRuns.slice(0, 10).map((run) => (
              <tr key={run.eval_run_id} onClick={() => navigate(`/eval/${run.eval_run_id}`)}>
                <td className="mono" style={{ color: 'var(--text-primary)' }}>{run.eval_run_id}</td>
                <td className="mono">{run.checkpoint_name ?? '-'}</td>
                <td>{run.benchmarks.map((b) => <span key={b} className="tag">{b}</span>)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

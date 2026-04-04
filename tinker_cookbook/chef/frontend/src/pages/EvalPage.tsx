import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import { SortableTable } from '../components/SortableTable';
import type { EvalRunSummary, ScoresTableRow } from '../api/types';

function scoreColor(score: number): string {
  if (score >= 0.8) return 'var(--success)';
  if (score >= 0.5) return 'var(--warning)';
  return 'var(--error)';
}

function ScoreCell({ score, prevScore }: { score?: number; prevScore?: number }) {
  if (score === undefined) return <span className="text-muted">-</span>;
  const delta = prevScore !== undefined ? score - prevScore : undefined;
  return (
    <span className="mono" style={{ color: scoreColor(score) }}>
      {(score * 100).toFixed(1)}%
      {delta !== undefined && delta !== 0 && (
        <span style={{
          fontSize: '0.625rem',
          marginLeft: '0.25rem',
          color: delta > 0 ? 'var(--success)' : 'var(--error)',
        }}>
          {delta > 0 ? '+' : ''}{(delta * 100).toFixed(1)}
        </span>
      )}
    </span>
  );
}

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

  const benchmarks = useMemo(() => {
    const all = new Set<string>();
    for (const row of scoresTable) {
      for (const key of Object.keys(row.scores)) all.add(key);
    }
    return [...all].sort();
  }, [scoresTable]);

  if (loading) return <div className="loading">Loading eval data...</div>;

  if (evalRuns.length === 0) {
    return (
      <div className="empty-state">
        <div style={{ fontSize: '1.25rem', marginBottom: '0.5rem', color: 'var(--text-primary)' }}>
          No Eval Data
        </div>
        <p>Run benchmarks with the eval framework and point tinker-chef at the eval store directory.</p>
      </div>
    );
  }

  const runColumns = [
    {
      key: 'run_id',
      label: 'Run',
      render: (r: EvalRunSummary) => <span className="mono" style={{ color: 'var(--text-primary)' }}>{r.eval_run_id}</span>,
      sortValue: (r: EvalRunSummary) => r.eval_run_id,
    },
    {
      key: 'model',
      label: 'Model',
      render: (r: EvalRunSummary) => <span>{r.model_name}</span>,
      sortValue: (r: EvalRunSummary) => r.model_name,
    },
    {
      key: 'checkpoint',
      label: 'Checkpoint',
      render: (r: EvalRunSummary) => <span className="mono">{r.checkpoint_name ?? '-'}</span>,
      sortValue: (r: EvalRunSummary) => r.checkpoint_name ?? '',
    },
    {
      key: 'benchmarks',
      label: 'Benchmarks',
      render: (r: EvalRunSummary) => (
        <>{r.benchmarks.map((b) => <span key={b} className="tag">{b}</span>)}</>
      ),
    },
    {
      key: 'timestamp',
      label: 'Time',
      render: (r: EvalRunSummary) => <span className="text-muted">{r.timestamp ?? '-'}</span>,
      sortValue: (r: EvalRunSummary) => r.timestamp ?? '',
    },
  ];

  return (
    <div>
      <h2 className="page-title">Evaluation Results</h2>
      <div className="page-subtitle">
        {evalRuns.length} eval run{evalRuns.length !== 1 ? 's' : ''} across {benchmarks.length} benchmark{benchmarks.length !== 1 ? 's' : ''}
      </div>

      {/* Scores matrix with deltas */}
      {scoresTable.length > 0 && benchmarks.length > 0 && (
        <div className="card" style={{ marginBottom: '1rem', overflow: 'auto' }}>
          <div className="card-header">
            <span className="card-title">Model Progression</span>
            <span style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', fontWeight: 400, textTransform: 'none', letterSpacing: 0 }}>
              Scores show delta from previous checkpoint
            </span>
          </div>
          <table>
            <thead>
              <tr>
                <th style={{ position: 'sticky', left: 0, background: 'var(--bg-card)', zIndex: 1 }}>Checkpoint</th>
                <th style={{ position: 'sticky', left: 0, background: 'var(--bg-card)' }}>Model</th>
                {benchmarks.map((b) => <th key={b}>{b}</th>)}
              </tr>
            </thead>
            <tbody>
              {scoresTable.map((row, rowIdx) => {
                const prevRow = rowIdx > 0 ? scoresTable[rowIdx - 1] : undefined;
                return (
                  <tr key={row.run_id} style={{ cursor: 'default' }}>
                    <td className="mono" style={{ fontWeight: 600, color: 'var(--text-primary)', position: 'sticky', left: 0, background: 'var(--bg-card)' }}>
                      {row.checkpoint_name ?? row.run_id}
                    </td>
                    <td style={{ position: 'sticky', left: 0, background: 'var(--bg-card)' }}>{row.model_name}</td>
                    {benchmarks.map((b) => (
                      <td key={b}>
                        <ScoreCell score={row.scores[b]} prevScore={prevRow?.scores[b]} />
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Eval runs list */}
      <div className="card" style={{ padding: 0, overflow: 'auto' }}>
        <SortableTable
          columns={runColumns}
          data={evalRuns}
          rowKey={(r) => r.eval_run_id}
          onRowClick={(r) => navigate(`/eval/${r.eval_run_id}`)}
        />
      </div>
    </div>
  );
}

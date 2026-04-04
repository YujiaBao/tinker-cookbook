import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import { StatusBadge, TypeBadge } from '../utils/shared';
import type { CheckpointRecord, MetricRecord, RunInfo, ScoresTableRow } from '../api/types';

interface Props {
  runId: string;
  run: RunInfo;
}

function pickPrimaryMetrics(keys: string[], trainingType: string | null): string[] {
  const priorities: Record<string, string[]> = {
    rl: ['env/all/reward/total', 'optim/kl_sample_train_v1', 'env/all/turns_per_episode', 'env/all/ac_tokens_per_turn'],
    sl: ['train_mean_nll', 'learning_rate', 'num_tokens', 'num_loss_tokens'],
    dpo: ['dpo_loss', 'accuracy', 'margin', 'chosen_reward'],
  };
  const preferred = priorities[trainingType ?? 'rl'] ?? priorities.rl;
  const found: string[] = [];
  for (const p of preferred) {
    if (keys.includes(p)) found.push(p);
    if (found.length >= 4) break;
  }
  // Fill with any remaining keys if needed
  if (found.length < 4) {
    for (const k of keys) {
      if (!found.includes(k) && !k.endsWith(':total') && !k.endsWith(':count') && !k.startsWith('time/')) {
        found.push(k);
        if (found.length >= 4) break;
      }
    }
  }
  return found;
}

function Sparkline({ data, color }: { data: number[]; color: string }) {
  if (data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const h = 32;
  const w = 120;
  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - ((v - min) / range) * (h - 4) - 2;
    return `${x},${y}`;
  }).join(' ');
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" />
    </svg>
  );
}

export function RunOverviewPanel({ runId, run }: Props) {
  const [metrics, setMetrics] = useState<MetricRecord[]>([]);
  const [keys, setKeys] = useState<string[]>([]);
  const [checkpoints, setCheckpoints] = useState<CheckpointRecord[]>([]);
  const [scores, setScores] = useState<ScoresTableRow[]>([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    Promise.all([
      api.getMetrics(runId),
      api.getMetricKeys(runId),
      api.getCheckpoints(runId).catch(() => []),
      api.getScoresTable().catch(() => []),
    ])
      .then(([metricsResp, metricKeys, ckpts, scoreTable]) => {
        setMetrics(metricsResp.records);
        setKeys(metricKeys);
        setCheckpoints(ckpts);
        setScores(scoreTable);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [runId]);

  if (loading) return <div className="loading">Loading overview...</div>;

  const primaryMetrics = pickPrimaryMetrics(keys, run.training_type ?? null);

  // Get last 30 values for sparklines
  const tail = metrics.slice(-30);

  return (
    <div>
      {/* Status + type header */}
      <div className="card" style={{ marginBottom: '0.75rem' }}>
        <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap', alignItems: 'flex-start' }}>
          {run.training_type && (
            <div>
              <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', marginBottom: '2px' }}>Type</div>
              <TypeBadge type={run.training_type} />
            </div>
          )}
          <div>
            <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', marginBottom: '2px' }}>Status</div>
            <StatusBadge status={run.status} />
          </div>
          {run.config_summary?.model_name != null && (
            <div>
              <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', marginBottom: '2px' }}>Model</div>
              <div className="mono" style={{ fontWeight: 600 }}>{String(run.config_summary.model_name)}</div>
            </div>
          )}
          <div>
            <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', marginBottom: '2px' }}>Steps</div>
            <div className="mono" style={{ fontWeight: 600 }}>{run.total_steps ?? metrics.length}</div>
          </div>
          {checkpoints.length > 0 && (
            <div>
              <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', marginBottom: '2px' }}>Checkpoints</div>
              <div className="mono" style={{ fontWeight: 600 }}>{checkpoints.length}</div>
            </div>
          )}
        </div>
      </div>

      {/* Key metrics sparklines */}
      {primaryMetrics.length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '0.625rem', marginBottom: '0.75rem' }}>
          {primaryMetrics.map((key) => {
            const values = tail.map((r) => r[key]).filter((v): v is number => typeof v === 'number');
            const latest = values[values.length - 1];
            const prev = values.length > 1 ? values[values.length - 2] : undefined;
            const shortName = key.includes('/') ? key.split('/').slice(1).join('/') : key;
            return (
              <div key={key} className="card" style={{ padding: '0.625rem 0.75rem' }}>
                <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>{shortName}</div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
                  <div>
                    <span className="mono" style={{ fontSize: '1.125rem', fontWeight: 700 }}>
                      {latest !== undefined ? (Math.abs(latest) < 0.01 || Math.abs(latest) > 999 ? latest.toExponential(2) : latest.toFixed(3)) : '-'}
                    </span>
                    {prev !== undefined && latest !== undefined && (
                      <span style={{ fontSize: '0.625rem', marginLeft: '0.25rem', color: latest > prev ? 'var(--success)' : latest < prev ? 'var(--error)' : 'var(--text-muted)' }}>
                        {latest > prev ? '\u2191' : latest < prev ? '\u2193' : ''}
                      </span>
                    )}
                  </div>
                  <Sparkline data={values} color="var(--accent)" />
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Checkpoint timeline with eval scores */}
      {checkpoints.length > 0 && (
        <div className="card">
          <div className="card-header">
            <span className="card-title">Checkpoint Timeline</span>
          </div>
          <table>
            <thead>
              <tr>
                <th>Checkpoint</th>
                <th>Step</th>
                <th>Type</th>
                {scores.length > 0 && <th>Eval Scores</th>}
              </tr>
            </thead>
            <tbody>
              {checkpoints.map((ckpt) => {
                const matchingScore = scores.find((s) =>
                  s.checkpoint_name === ckpt.name || s.run_id.includes(ckpt.name)
                );
                return (
                  <tr key={ckpt.name} style={{ cursor: matchingScore ? 'pointer' : 'default' }}
                    onClick={() => matchingScore && navigate(`/eval/${matchingScore.run_id}`)}>
                    <td className="mono" style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                      {ckpt.name}
                      {ckpt.kind === 'final' && <span className="badge badge-green" style={{ marginLeft: '0.375rem' }}>final</span>}
                    </td>
                    <td className="mono">{ckpt.loop_state?.batch ?? '-'}</td>
                    <td>
                      <span className="tag">{ckpt.kind}</span>
                    </td>
                    {scores.length > 0 && (
                      <td>
                        {matchingScore ? (
                          Object.entries(matchingScore.scores).map(([bench, score]) => (
                            <span key={bench} className="mono" style={{ marginRight: '0.75rem', fontSize: '0.75rem' }}>
                              <span className="text-muted">{bench}: </span>
                              <span style={{ color: score >= 0.8 ? 'var(--success)' : score >= 0.5 ? 'var(--warning)' : 'var(--error)' }}>
                                {(score * 100).toFixed(1)}%
                              </span>
                            </span>
                          ))
                        ) : <span className="text-muted">-</span>}
                      </td>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {metrics.length === 0 && checkpoints.length === 0 && (
        <div className="empty-state">
          <p>No data yet. Metrics will appear as training progresses.</p>
        </div>
      )}
    </div>
  );
}

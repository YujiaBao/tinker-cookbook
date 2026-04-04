import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { api } from '../api/client';
import type { EvalTrajectoryDetail } from '../api/types';

export function EvalTrajectoryPage() {
  const { evalRunId, benchmark, idx } = useParams<{
    evalRunId: string;
    benchmark: string;
    idx: string;
  }>();
  const [traj, setTraj] = useState<EvalTrajectoryDetail | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!evalRunId || !benchmark || !idx) return;
    api
      .getEvalTrajectoryDetail(evalRunId, benchmark, Number(idx))
      .then(setTraj)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [evalRunId, benchmark, idx]);

  if (loading) return <div className="loading">Loading trajectory...</div>;
  if (!traj) return <div className="empty-state">Trajectory not found</div>;

  return (
    <div>
      <div className="breadcrumb">
        <Link to="/eval">Eval</Link>
        <span>/</span>
        <Link to={`/eval/${evalRunId}`}>{evalRunId}</Link>
        <span>/</span>
        <span>{benchmark}</span>
        <span>/</span>
        <span>#{idx}</span>
      </div>

      {/* Header */}
      <div className="card" style={{ marginBottom: '16px' }}>
        <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
          <MetaField label="Benchmark" value={traj.benchmark} />
          <MetaField label="Index" value={String(traj.idx)} />
          {traj.example_id && (
            <MetaField label="Example ID" value={traj.example_id.slice(0, 16)} />
          )}
          <MetaField
            label="Reward"
            value={traj.reward.toFixed(2)}
            color={traj.reward > 0 ? 'var(--success)' : 'var(--error)'}
          />
          <MetaField label="Turns" value={String(traj.turns.length)} />
          <MetaField label="Time" value={`${traj.time_seconds.toFixed(1)}s`} />
          {traj.error && <MetaField label="Error" value={traj.error} color="var(--error)" />}
        </div>

        {/* Logs */}
        {Object.keys(traj.logs).length > 0 && (
          <div style={{ marginTop: '12px', padding: '8px 12px', background: 'var(--bg-tertiary)', borderRadius: '6px' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '4px', fontWeight: 600 }}>
              LOGS
            </div>
            <div style={{ fontSize: '0.8rem' }}>
              {Object.entries(traj.logs).map(([k, v]) => (
                <div key={k} style={{ marginBottom: '2px' }}>
                  <span className="text-muted">{k}:</span>{' '}
                  <span className="mono">{String(v)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Conversation */}
      <div className="card">
        <div className="card-header">Conversation ({traj.turns.length} turns)</div>
        <div className="conversation">
          {traj.turns.map((turn, i) => (
            <div key={i} className={`message message-${turn.role}`}>
              <div className="message-role">
                {turn.role}
                <span style={{ fontSize: '0.65rem', fontWeight: 400, marginLeft: '8px', color: 'var(--text-muted)' }}>
                  {turn.token_count} tokens
                </span>
              </div>
              <div className="message-content">{turn.content}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function MetaField({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div>
      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '2px' }}>{label}</div>
      <div className="mono" style={{ fontWeight: 600, color: color ?? 'var(--text-primary)' }}>{value}</div>
    </div>
  );
}

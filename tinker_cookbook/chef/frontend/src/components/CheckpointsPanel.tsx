import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { CheckpointRecord } from '../api/types';

interface Props {
  runId: string;
}

export function CheckpointsPanel({ runId }: Props) {
  const [checkpoints, setCheckpoints] = useState<CheckpointRecord[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getCheckpoints(runId)
      .then(setCheckpoints)
      .catch(() => setCheckpoints([]))
      .finally(() => setLoading(false));
  }, [runId]);

  if (loading) return <div className="loading">Loading checkpoints...</div>;
  if (checkpoints.length === 0) return <div className="empty-state">No checkpoints saved yet</div>;

  // Separate full checkpoints from rolling checkpoints
  const fullCkpts = checkpoints.filter((c) => !isRolling(c));
  const rollingCkpts = checkpoints.filter((c) => isRolling(c));

  return (
    <div>
      {/* Full checkpoints */}
      <div className="card" style={{ marginBottom: '0.75rem' }}>
        <div className="card-header">
          <span className="card-title">Checkpoints ({fullCkpts.length})</span>
        </div>
        {fullCkpts.length > 0 ? (
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Step</th>
                <th>Type</th>
                <th>State Path</th>
                <th>Sampler Path</th>
              </tr>
            </thead>
            <tbody>
              {fullCkpts.map((ckpt) => (
                <tr key={ckpt.name} style={{ cursor: 'default' }}>
                  <td className="mono" style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                    {ckpt.name}
                    {ckpt.kind === 'final' && (
                      <span className="badge badge-green" style={{ marginLeft: '0.375rem' }}>final</span>
                    )}
                  </td>
                  <td className="mono">{ckpt.loop_state?.batch ?? ckpt.loop_state?.epoch ?? '-'}</td>
                  <td>
                    <span className="tag">
                      {ckpt.kind === 'final' ? 'final' : ckpt.kind === 'both' ? 'full' : ckpt.kind ?? 'checkpoint'}
                    </span>
                  </td>
                  <td className="mono" style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', maxWidth: '300px', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {ckpt.state_path ?? '-'}
                  </td>
                  <td className="mono" style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', maxWidth: '300px', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {ckpt.sampler_weights_path ?? '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="text-muted" style={{ padding: '0.5rem', fontSize: '0.8125rem' }}>No full checkpoints</div>
        )}
      </div>

      {/* Rolling checkpoints */}
      {rollingCkpts.length > 0 && (
        <div className="card">
          <div className="card-header">
            <span className="card-title">Rolling Checkpoints ({rollingCkpts.length})</span>
            <span className="text-muted" style={{ fontSize: '0.625rem', fontWeight: 400, textTransform: 'none', letterSpacing: 0 }}>
              Resume-only, short TTL
            </span>
          </div>
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Step</th>
                <th>State Path</th>
              </tr>
            </thead>
            <tbody>
              {rollingCkpts.map((ckpt) => (
                <tr key={ckpt.name} style={{ cursor: 'default' }}>
                  <td className="mono">{ckpt.name}</td>
                  <td className="mono">{ckpt.loop_state?.batch ?? '-'}</td>
                  <td className="mono" style={{ fontSize: '0.6875rem', color: 'var(--text-muted)' }}>
                    {ckpt.state_path ?? '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function isRolling(ckpt: CheckpointRecord): boolean {
  // Rolling checkpoints typically have state_path but no sampler_path
  return ckpt.state_path != null && !ckpt.sampler_weights_path && ckpt.kind !== 'final';
}

import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import type { IterationInfo, RolloutSummary } from '../api/types';

interface Props {
  runId: string;
  iterations: IterationInfo[];
}

function rewardClass(reward: number): string {
  if (reward >= 0.8) return 'high';
  if (reward >= 0.3) return 'mid';
  return 'low';
}

export function RolloutBrowser({ runId, iterations }: Props) {
  const navigate = useNavigate();
  const [selectedIter, setSelectedIter] = useState<number | null>(null);
  const [rollouts, setRollouts] = useState<RolloutSummary[]>([]);
  const [availableTags, setAvailableTags] = useState<string[]>([]);
  const [tagFilter, setTagFilter] = useState<string>('');
  const [loading, setLoading] = useState(false);

  // Select the first iteration with rollouts by default
  useEffect(() => {
    const withRollouts = iterations.filter((it) => it.has_train_rollouts);
    if (withRollouts.length > 0 && selectedIter === null) {
      setSelectedIter(withRollouts[0].iteration);
    }
  }, [iterations, selectedIter]);

  // Fetch rollouts when iteration or filter changes
  useEffect(() => {
    if (selectedIter === null) return;
    setLoading(true);
    api
      .getRollouts(runId, selectedIter, {
        tag: tagFilter || undefined,
      })
      .then((resp) => {
        setRollouts(resp.rollouts);
        setAvailableTags(resp.available_tags);
      })
      .catch(() => setRollouts([]))
      .finally(() => setLoading(false));
  }, [runId, selectedIter, tagFilter]);

  if (iterations.length === 0) {
    return <div className="empty-state">No iteration data available</div>;
  }

  const iterationsWithRollouts = iterations.filter((it) => it.has_train_rollouts);

  return (
    <div>
      <div className="filters-bar">
        <div className="filter-group">
          <span className="filter-label">Iteration</span>
          <select
            value={selectedIter ?? ''}
            onChange={(e) => setSelectedIter(Number(e.target.value))}
          >
            {iterationsWithRollouts.map((it) => (
              <option key={it.iteration} value={it.iteration}>
                {it.iteration}
              </option>
            ))}
          </select>
        </div>

        {availableTags.length > 0 && (
          <div className="filter-group">
            <span className="filter-label">Tag</span>
            <select value={tagFilter} onChange={(e) => setTagFilter(e.target.value)}>
              <option value="">All</option>
              {availableTags.map((tag) => (
                <option key={tag} value={tag}>{tag}</option>
              ))}
            </select>
          </div>
        )}

        <div style={{ marginLeft: 'auto', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          {rollouts.length} rollout{rollouts.length !== 1 ? 's' : ''}
        </div>
      </div>

      {loading ? (
        <div className="loading">Loading rollouts...</div>
      ) : rollouts.length === 0 ? (
        <div className="empty-state">No rollouts for this iteration</div>
      ) : (
        <div className="card" style={{ padding: 0, overflow: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Group</th>
                <th>Traj</th>
                <th>Tags</th>
                <th>Steps</th>
                <th>Total Reward</th>
                <th>Final Reward</th>
                <th>Context Len</th>
              </tr>
            </thead>
            <tbody>
              {rollouts.map((r) => (
                <tr
                  key={`${r.group_idx}-${r.traj_idx}`}
                  onClick={() =>
                    navigate(
                      `/runs/${runId}/iterations/${selectedIter}/rollouts/${r.group_idx}/${r.traj_idx}`
                    )
                  }
                >
                  <td className="mono">{r.group_idx}</td>
                  <td className="mono">{r.traj_idx}</td>
                  <td>
                    {r.tags.map((tag) => (
                      <span key={tag} className="tag">{tag}</span>
                    ))}
                  </td>
                  <td className="mono">{r.num_steps}</td>
                  <td>
                    <span className={`reward-badge ${rewardClass(r.total_reward)}`}>
                      {r.total_reward.toFixed(3)}
                    </span>
                  </td>
                  <td>
                    <span className={`reward-badge ${rewardClass(r.final_reward)}`}>
                      {r.final_reward.toFixed(3)}
                    </span>
                  </td>
                  <td className="mono">{r.final_ob_len}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import { SortableTable } from './SortableTable';
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
  const [minReward, setMinReward] = useState<string>('');
  const [maxReward, setMaxReward] = useState<string>('');
  const [loading, setLoading] = useState(false);

  const iterationsWithRollouts = iterations.filter((it) => it.has_train_rollouts);

  useEffect(() => {
    if (iterationsWithRollouts.length > 0 && selectedIter === null) {
      setSelectedIter(iterationsWithRollouts[0].iteration);
    }
  }, [iterations, selectedIter]);

  useEffect(() => {
    if (selectedIter === null) return;
    setLoading(true);
    api
      .getRollouts(runId, selectedIter, {
        tag: tagFilter || undefined,
        min_reward: minReward !== '' ? Number(minReward) : undefined,
        max_reward: maxReward !== '' ? Number(maxReward) : undefined,
      })
      .then((resp) => {
        setRollouts(resp.rollouts);
        setAvailableTags(resp.available_tags);
      })
      .catch(() => setRollouts([]))
      .finally(() => setLoading(false));
  }, [runId, selectedIter, tagFilter, minReward, maxReward]);

  if (iterations.length === 0) {
    return <div className="empty-state">No iteration data available</div>;
  }

  const columns = [
    {
      key: 'group',
      label: 'Group',
      render: (r: RolloutSummary) => <span className="mono">{r.group_idx}</span>,
      sortValue: (r: RolloutSummary) => r.group_idx,
    },
    {
      key: 'traj',
      label: 'Traj',
      render: (r: RolloutSummary) => <span className="mono">{r.traj_idx}</span>,
      sortValue: (r: RolloutSummary) => r.traj_idx,
    },
    {
      key: 'tags',
      label: 'Tags',
      render: (r: RolloutSummary) => (
        <>{r.tags.map((tag) => <span key={tag} className="tag">{tag}</span>)}</>
      ),
    },
    {
      key: 'steps',
      label: 'Steps',
      render: (r: RolloutSummary) => <span className="mono">{r.num_steps}</span>,
      sortValue: (r: RolloutSummary) => r.num_steps,
    },
    {
      key: 'total_reward',
      label: 'Total Reward',
      render: (r: RolloutSummary) => (
        <span className={`reward-badge ${rewardClass(r.total_reward)}`}>
          {r.total_reward.toFixed(3)}
        </span>
      ),
      sortValue: (r: RolloutSummary) => r.total_reward,
    },
    {
      key: 'final_reward',
      label: 'Final Reward',
      render: (r: RolloutSummary) => (
        <span className={`reward-badge ${rewardClass(r.final_reward)}`}>
          {r.final_reward.toFixed(3)}
        </span>
      ),
      sortValue: (r: RolloutSummary) => r.final_reward,
    },
    {
      key: 'context',
      label: 'Context',
      render: (r: RolloutSummary) => <span className="mono">{r.final_ob_len}</span>,
      sortValue: (r: RolloutSummary) => r.final_ob_len,
    },
  ];

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

        <div className="filter-group">
          <span className="filter-label">Reward</span>
          <input
            type="number"
            placeholder="min"
            value={minReward}
            onChange={(e) => setMinReward(e.target.value)}
            step="0.1"
            style={{
              width: '60px',
              padding: '0.3125rem 0.375rem',
              borderRadius: '6px',
              border: '1px solid var(--border)',
              background: 'var(--bg-tertiary)',
              color: 'var(--text-primary)',
              fontSize: '0.8125rem',
            }}
          />
          <span className="text-muted" style={{ fontSize: '0.75rem' }}>to</span>
          <input
            type="number"
            placeholder="max"
            value={maxReward}
            onChange={(e) => setMaxReward(e.target.value)}
            step="0.1"
            style={{
              width: '60px',
              padding: '0.3125rem 0.375rem',
              borderRadius: '6px',
              border: '1px solid var(--border)',
              background: 'var(--bg-tertiary)',
              color: 'var(--text-primary)',
              fontSize: '0.8125rem',
            }}
          />
        </div>

        <div style={{ marginLeft: 'auto', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          {rollouts.length} rollout{rollouts.length !== 1 ? 's' : ''}
        </div>
      </div>

      {loading ? (
        <div className="loading">Loading rollouts...</div>
      ) : (
        <div className="card" style={{ padding: 0, overflow: 'auto' }}>
          <SortableTable
            columns={columns}
            data={rollouts}
            rowKey={(r) => `${r.group_idx}-${r.traj_idx}`}
            onRowClick={(r) =>
              navigate(`/runs/${runId}/iterations/${selectedIter}/rollouts/${r.group_idx}/${r.traj_idx}`)
            }
          />
        </div>
      )}
    </div>
  );
}

import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { api } from '../api/client';
import type { RolloutDetail, LogtreeNode, LogtreeResponse } from '../api/types';

export function RolloutDetailPage() {
  const { runId, iteration, groupIdx, trajIdx } = useParams<{
    runId: string;
    iteration: string;
    groupIdx: string;
    trajIdx: string;
  }>();
  const [rollout, setRollout] = useState<RolloutDetail | null>(null);
  const [logtree, setLogtree] = useState<LogtreeResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!runId || !iteration || !groupIdx || !trajIdx) return;
    const iter = Number(iteration);
    const gIdx = Number(groupIdx);
    const tIdx = Number(trajIdx);

    Promise.all([
      api.getRolloutDetail(runId, iter, gIdx, tIdx),
      api.getLogtree(runId, iter).catch(() => null),
    ])
      .then(([rolloutData, logtreeData]) => {
        setRollout(rolloutData);
        setLogtree(logtreeData);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [runId, iteration, groupIdx, trajIdx]);

  if (loading) return <div className="loading">Loading rollout...</div>;
  if (error) return <div className="empty-state">{error}</div>;
  if (!rollout || !runId) return <div className="empty-state">Rollout not found</div>;

  return (
    <div>
      <div className="breadcrumb">
        <Link to="/">Runs</Link>
        <span>/</span>
        <Link to={`/runs/${runId}`}>{runId}</Link>
        <span>/</span>
        <span>Iteration {iteration}</span>
        <span>/</span>
        <span>Rollout ({groupIdx}, {trajIdx})</span>
      </div>

      {/* Header with rollout metadata */}
      <div className="card" style={{ marginBottom: '16px' }}>
        <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
          <MetaField label="Iteration" value={String(rollout.iteration)} />
          <MetaField label="Group" value={String(rollout.group_idx)} />
          <MetaField label="Trajectory" value={String(rollout.traj_idx)} />
          <MetaField
            label="Total Reward"
            value={rollout.total_reward.toFixed(3)}
            color={rewardColor(rollout.total_reward)}
          />
          <MetaField
            label="Final Reward"
            value={rollout.final_reward.toFixed(3)}
            color={rewardColor(rollout.final_reward)}
          />
          <MetaField label="Steps" value={String(rollout.steps.length)} />
          <MetaField label="Final Context" value={`${rollout.final_ob_len} tokens`} />
          {rollout.tags.length > 0 && (
            <div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '2px' }}>
                Tags
              </div>
              <div>
                {rollout.tags.map((tag) => (
                  <span key={tag} className="tag">{tag}</span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Per-step timeline */}
      <div className="card" style={{ marginBottom: '16px' }}>
        <div className="card-title" style={{ marginBottom: '12px' }}>Step Timeline</div>
        <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
          {rollout.steps.map((step) => (
            <div
              key={step.step_idx}
              title={`Step ${step.step_idx}: reward=${step.reward}, ob=${step.ob_len}, ac=${step.ac_len}`}
              style={{
                flex: '0 0 auto',
                padding: '4px 8px',
                borderRadius: '4px',
                background: step.episode_done
                  ? 'rgba(34, 197, 94, 0.15)'
                  : 'var(--bg-tertiary)',
                border: `1px solid ${step.episode_done ? 'var(--success)' : 'var(--border)'}`,
                fontSize: '0.7rem',
                textAlign: 'center',
              }}
            >
              <div className="mono" style={{ fontWeight: 600 }}>
                {step.step_idx}
              </div>
              <div className="mono" style={{ color: rewardColor(step.reward) }}>
                r={step.reward.toFixed(2)}
              </div>
              <div className="text-muted" style={{ fontSize: '0.65rem' }}>
                {step.ob_len}+{step.ac_len}t
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Conversation from logtree (if available) */}
      {logtree && (
        <div className="card" style={{ marginBottom: '16px' }}>
          <div className="card-title" style={{ marginBottom: '12px' }}>Conversation</div>
          <LogtreeConversation node={logtree.root} />
        </div>
      )}

      {/* Step details table */}
      <div className="card">
        <div className="card-title" style={{ marginBottom: '12px' }}>Step Details</div>
        <table className="data-table">
          <thead>
            <tr>
              <th>Step</th>
              <th>Obs Tokens</th>
              <th>Action Tokens</th>
              <th>Reward</th>
              <th>Done</th>
              <th>Metrics</th>
            </tr>
          </thead>
          <tbody>
            {rollout.steps.map((step) => (
              <tr key={step.step_idx} style={{ cursor: 'default' }}>
                <td className="mono">{step.step_idx}</td>
                <td className="mono">{step.ob_len}</td>
                <td className="mono">{step.ac_len}</td>
                <td>
                  <span style={{ color: rewardColor(step.reward), fontWeight: 600 }}>
                    {step.reward.toFixed(3)}
                  </span>
                </td>
                <td>{step.episode_done ? 'Yes' : ''}</td>
                <td style={{ fontSize: '0.75rem' }}>
                  {Object.entries(step.metrics)
                    .map(([k, v]) => `${k}=${typeof v === 'number' ? v.toFixed(3) : v}`)
                    .join(', ') || '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function MetaField({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div>
      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '2px' }}>
        {label}
      </div>
      <div className="mono" style={{ fontWeight: 600, color: color ?? 'var(--text-primary)' }}>
        {value}
      </div>
    </div>
  );
}

function rewardColor(reward: number): string {
  if (reward >= 0.8) return 'var(--reward-high)';
  if (reward >= 0.3) return 'var(--reward-mid)';
  return 'var(--reward-low)';
}

/** Recursively render logtree nodes, extracting conversations. */
function LogtreeConversation({ node }: { node: LogtreeNode }) {
  // If this node has conversation data, render it
  if (node.data && (node.data as { type?: string }).type === 'conversation') {
    const messages = (node.data as { messages?: ConvMessage[] }).messages ?? [];
    return (
      <div className="conversation">
        {messages.map((msg, i) => (
          <ConversationMessage key={i} message={msg} />
        ))}
      </div>
    );
  }

  // Otherwise, recurse into children
  if (!node.children) return null;
  return (
    <>
      {node.children.map((child, i) => {
        if (typeof child === 'string') {
          return child.trim() ? (
            <div key={i} style={{ padding: '4px 0', color: 'var(--text-secondary)' }}>
              {child}
            </div>
          ) : null;
        }
        return <LogtreeConversation key={i} node={child} />;
      })}
    </>
  );
}

interface ConvMessage {
  role: string;
  content: string | ContentPart[];
}

interface ContentPart {
  type: string;
  text?: string;
  thinking?: string;
  tool_call?: { function: { name: string; arguments: string } };
  raw_text?: string;
  error?: string;
}

function ConversationMessage({ message }: { message: ConvMessage }) {
  const [thinkingOpen, setThinkingOpen] = useState(false);

  return (
    <div className={`message role-${message.role}`}>
      <div className="message-role">{message.role}</div>
      <div className="message-content">
        {typeof message.content === 'string' ? (
          message.content
        ) : (
          message.content.map((part, i) => {
            if (part.type === 'text') {
              return <span key={i}>{part.text}</span>;
            }
            if (part.type === 'thinking' && part.thinking) {
              return (
                <div key={i} className="thinking-block">
                  <div className="thinking-toggle" onClick={() => setThinkingOpen(!thinkingOpen)}>
                    {thinkingOpen ? '▼' : '▶'} Thinking
                  </div>
                  {thinkingOpen && <div className="thinking-content">{part.thinking}</div>}
                </div>
              );
            }
            if (part.type === 'tool_call' && part.tool_call) {
              return (
                <div key={i} className="tool-call-block">
                  <div className="tool-call-label">Tool Call: {part.tool_call.function.name}</div>
                  <pre className="tool-call-code">{part.tool_call.function.arguments}</pre>
                </div>
              );
            }
            if (part.type === 'unparsed_tool_call') {
              return (
                <div key={i} className="tool-call-block" style={{ borderColor: 'var(--error)' }}>
                  <div className="tool-call-label" style={{ color: 'var(--error)' }}>
                    Unparsed Tool Call
                  </div>
                  <pre className="tool-call-code">{part.raw_text}</pre>
                  {part.error && (
                    <div style={{ color: 'var(--error)', fontSize: '0.75rem', marginTop: '4px' }}>
                      {part.error}
                    </div>
                  )}
                </div>
              );
            }
            if (part.type === 'image') {
              return (
                <span key={i} className="tag">
                  [Image]
                </span>
              );
            }
            return null;
          })
        )}
      </div>
    </div>
  );
}

import { useEffect, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { api } from '../api/client';
import { MetaField, rewardColor } from '../utils/shared';
import type { RolloutDetail, RolloutSummary, LogtreeNode, LogtreeResponse } from '../api/types';

export function RolloutDetailPage() {
  const { runId, iteration, groupIdx, trajIdx } = useParams<{
    runId: string;
    iteration: string;
    groupIdx: string;
    trajIdx: string;
  }>();
  const navigate = useNavigate();
  const [rollout, setRollout] = useState<RolloutDetail | null>(null);
  const [logtree, setLogtree] = useState<LogtreeResponse | null>(null);
  const [siblings, setSiblings] = useState<RolloutSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!runId || !iteration || !groupIdx || !trajIdx) return;
    const iter = Number(iteration);
    const gIdx = Number(groupIdx);
    const tIdx = Number(trajIdx);

    setLoading(true);
    Promise.all([
      api.getRolloutDetail(runId, iter, gIdx, tIdx),
      api.getLogtree(runId, iter).catch(() => null),
      api.getRollouts(runId, iter),
    ])
      .then(([rolloutData, logtreeData, rolloutsResp]) => {
        setRollout(rolloutData);
        setLogtree(logtreeData);
        setSiblings(rolloutsResp.rollouts);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [runId, iteration, groupIdx, trajIdx]);

  if (loading) return <div className="loading">Loading rollout...</div>;
  if (error) return <div className="empty-state">{error}</div>;
  if (!rollout || !runId) return <div className="empty-state">Rollout not found</div>;

  // Find current position among siblings for prev/next
  const currentIdx = siblings.findIndex(
    (s) => s.group_idx === rollout.group_idx && s.traj_idx === rollout.traj_idx
  );
  const prevSibling = currentIdx > 0 ? siblings[currentIdx - 1] : null;
  const nextSibling = currentIdx >= 0 && currentIdx < siblings.length - 1 ? siblings[currentIdx + 1] : null;

  const navTo = (s: RolloutSummary) =>
    `/runs/${runId}/iterations/${iteration}/rollouts/${s.group_idx}/${s.traj_idx}`;

  return (
    <div>
      {/* Quick-nav back to run tabs */}
      <div className="tabs" style={{ marginBottom: '0.5rem' }}>
        <Link to={`/runs/${runId}`} className="tab">Overview</Link>
        <Link to={`/runs/${runId}`} className="tab">Metrics</Link>
        <span className="tab active">Rollout</span>
        <Link to={`/runs/${runId}`} className="tab">Checkpoints</Link>
        <Link to={`/runs/${runId}`} className="tab">Timing</Link>
        <Link to={`/runs/${runId}`} className="tab">Config</Link>
      </div>

      {/* Breadcrumb + prev/next */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div className="breadcrumb" style={{ marginBottom: 0 }}>
          <Link to="/">Dashboard</Link>
          <span>/</span>
          <Link to={`/runs/${runId}`}>{runId}</Link>
          <span>/</span>
          <span>Iter {iteration}</span>
          <span>/</span>
          <span>({groupIdx}, {trajIdx})</span>
        </div>
        <div style={{ display: 'flex', gap: '0.375rem', alignItems: 'center' }}>
          {currentIdx >= 0 && (
            <span className="text-muted" style={{ fontSize: '0.6875rem', marginRight: '0.25rem' }}>
              {currentIdx + 1} of {siblings.length}
            </span>
          )}
          <button
            className="tab"
            onClick={() => prevSibling && navigate(navTo(prevSibling))}
            disabled={!prevSibling}
            style={{
              padding: '0.25rem 0.5rem',
              fontSize: '0.75rem',
              opacity: prevSibling ? 1 : 0.3,
              borderBottom: 'none',
            }}
          >
            Prev
          </button>
          <button
            className="tab"
            onClick={() => nextSibling && navigate(navTo(nextSibling))}
            disabled={!nextSibling}
            style={{
              padding: '0.25rem 0.5rem',
              fontSize: '0.75rem',
              opacity: nextSibling ? 1 : 0.3,
              borderBottom: 'none',
            }}
          >
            Next
          </button>
        </div>
      </div>

      {/* Header with rollout metadata */}
      <div className="card" style={{ marginBottom: '0.75rem' }}>
        <div style={{ display: 'flex', gap: '1.25rem', flexWrap: 'wrap', alignItems: 'flex-start' }}>
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
          <MetaField label="Context" value={`${rollout.final_ob_len} tok`} />
          {rollout.sampling_client_step != null && (
            <MetaField label="Sampled At" value={`step ${rollout.sampling_client_step}`} />
          )}
          {rollout.tags.length > 0 && (
            <div>
              <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', marginBottom: '2px' }}>Tags</div>
              <div>{rollout.tags.map((tag) => <span key={tag} className="tag">{tag}</span>)}</div>
            </div>
          )}
        </div>
      </div>

      {/* Step timeline */}
      <div className="card" style={{ marginBottom: '0.75rem' }}>
        <div className="card-title" style={{ marginBottom: '0.5rem' }}>Step Timeline</div>
        <div style={{ display: 'flex', gap: '3px', flexWrap: 'wrap' }}>
          {rollout.steps.map((step) => (
            <div
              key={step.step_idx}
              title={`Step ${step.step_idx}: reward=${step.reward}, ob=${step.ob_len}, ac=${step.ac_len}`}
              style={{
                flex: '0 0 auto',
                padding: '3px 6px',
                borderRadius: '4px',
                background: step.episode_done ? 'rgba(34, 197, 94, 0.15)' : 'var(--bg-tertiary)',
                border: `1px solid ${step.episode_done ? 'var(--success)' : 'var(--border)'}`,
                fontSize: '0.625rem',
                textAlign: 'center',
                minWidth: '42px',
              }}
            >
              <div className="mono" style={{ fontWeight: 600 }}>{step.step_idx}</div>
              <div className="mono" style={{ color: rewardColor(step.reward) }}>
                r={step.reward.toFixed(2)}
              </div>
              <div className="text-muted" style={{ fontSize: '0.5625rem' }}>
                {step.ob_len}+{step.ac_len}t
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Conversation from logtree — clean chat layout */}
      {logtree && (() => {
        const allMessages = extractAllMessages(logtree.root);
        if (allMessages.length === 0) return null;
        return (
          <div className="card" style={{ marginBottom: '0.75rem' }}>
            <div className="card-title" style={{ marginBottom: '0.5rem' }}>
              Conversation ({allMessages.length} messages)
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {allMessages.map((msg, i) => (
                <ChatBubble key={i} message={msg} />
              ))}
            </div>
          </div>
        );
      })()}

      {/* Step details table */}
      <div className="card">
        <div className="card-title" style={{ marginBottom: '0.5rem' }}>Step Details</div>
        <table>
          <thead>
            <tr>
              <th>Step</th>
              <th>Obs</th>
              <th>Action</th>
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
                <td style={{ fontSize: '0.6875rem' }}>
                  {Object.entries(step.metrics)
                    .map(([k, v]) => `${k}=${typeof v === 'number' ? v.toFixed(3) : v}`)
                    .join(', ') || '-'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

interface ConvMessage { role: string; content: string | ContentPart[]; }
interface ContentPart {
  type: string;
  text?: string;
  thinking?: string;
  tool_call?: { function: { name: string; arguments: string } };
  raw_text?: string;
  error?: string;
}

/** Recursively extract all conversation messages from a logtree, ignoring boilerplate. */
function extractAllMessages(node: LogtreeNode): ConvMessage[] {
  const messages: ConvMessage[] = [];
  if (node.data && (node.data as { type?: string }).type === 'conversation') {
    const msgs = (node.data as { messages?: ConvMessage[] }).messages ?? [];
    messages.push(...msgs);
  }
  if (node.children) {
    for (const child of node.children) {
      if (typeof child !== 'string') {
        messages.push(...extractAllMessages(child));
      }
    }
  }
  return messages;
}

/** Chat bubble: user messages on left, assistant on right. */
function ChatBubble({ message }: { message: ConvMessage }) {
  const [thinkingOpen, setThinkingOpen] = useState(false);
  const isUser = message.role === 'user' || message.role === 'system';
  const isAssistant = message.role === 'assistant';

  const roleColors: Record<string, string> = {
    user: 'var(--cyan)',
    assistant: 'var(--purple)',
    system: 'var(--warning)',
    tool: 'var(--accent)',
    environment: 'var(--accent)',
  };
  const color = roleColors[message.role] ?? 'var(--text-muted)';

  return (
    <div style={{
      display: 'flex',
      justifyContent: isAssistant ? 'flex-end' : 'flex-start',
    }}>
      <div style={{
        maxWidth: '80%',
        padding: '0.5rem 0.75rem',
        borderRadius: isUser ? '12px 12px 12px 2px' : '12px 12px 2px 12px',
        background: isAssistant ? 'var(--bg-elevated)' : 'var(--bg-surface)',
        border: '1px solid var(--border)',
      }}>
        <div style={{
          fontFamily: 'var(--font-mono)', fontSize: '0.5625rem', fontWeight: 600,
          color, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.25rem',
        }}>
          {message.role}
        </div>
        <div style={{ fontSize: '0.8125rem', lineHeight: 1.6, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
          {typeof message.content === 'string' ? (
            message.content
          ) : (
            message.content.map((part, i) => {
              if (part.type === 'text') return <span key={i}>{part.text}</span>;
              if (part.type === 'thinking' && part.thinking) {
                return (
                  <div key={i} className="thinking-block">
                    <div className="thinking-toggle" onClick={() => setThinkingOpen(!thinkingOpen)}>
                      {thinkingOpen ? '\u25bc' : '\u25b6'} Thinking
                    </div>
                    {thinkingOpen && <div className="thinking-content">{part.thinking}</div>}
                  </div>
                );
              }
              if (part.type === 'tool_call' && part.tool_call) {
                return (
                  <div key={i} className="tool-call-block">
                    <div className="tool-call-label">Tool: {part.tool_call.function.name}</div>
                    <pre className="tool-call-code">{part.tool_call.function.arguments}</pre>
                  </div>
                );
              }
              if (part.type === 'image') return <span key={i} className="tag">[Image]</span>;
              return null;
            })
          )}
        </div>
      </div>
    </div>
  );
}

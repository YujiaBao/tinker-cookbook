/** Shared UI utilities and constants used across multiple pages/components. */

export const TYPE_LABELS: Record<string, string> = { rl: 'RL', sl: 'SFT', dpo: 'DPO' };
export const TYPE_COLORS: Record<string, string> = { rl: '#6366f1', sl: '#22c55e', dpo: '#f59e0b' };

export function scoreColor(score: number): string {
  if (score >= 0.8) return 'var(--success)';
  if (score >= 0.5) return 'var(--warning)';
  return 'var(--error)';
}

export function rewardColor(reward: number): string {
  if (reward >= 0.8) return 'var(--reward-high)';
  if (reward >= 0.3) return 'var(--reward-mid)';
  return 'var(--reward-low)';
}

export function timeAgo(ts: number): string {
  const seconds = Math.floor(Date.now() / 1000 - ts);
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

const STATUS_STYLES: Record<string, { bg: string; fg: string }> = {
  running: { bg: 'rgba(34, 197, 94, 0.15)', fg: 'var(--success)' },
  completed: { bg: 'rgba(99, 102, 241, 0.15)', fg: '#818cf8' },
  idle: { bg: 'rgba(100, 116, 139, 0.15)', fg: 'var(--text-muted)' },
};

export function StatusBadge({ status }: { status: string }) {
  const c = STATUS_STYLES[status] ?? STATUS_STYLES.idle;
  return (
    <span className="badge" style={{ background: c.bg, color: c.fg }}>
      {status === 'running' && (
        <span style={{ width: 6, height: 6, borderRadius: '50%', background: c.fg, animation: 'pulse 2s infinite' }} />
      )}
      {status}
    </span>
  );
}

export function MetaField({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div>
      <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', marginBottom: '2px' }}>{label}</div>
      <div className="mono" style={{ fontWeight: 600, color: color ?? 'var(--text-primary)' }}>{value}</div>
    </div>
  );
}

export function TypeBadge({ type }: { type: string | null }) {
  if (!type) return null;
  return (
    <span className="tag" style={{ background: `${TYPE_COLORS[type]}22`, color: TYPE_COLORS[type] }}>
      {TYPE_LABELS[type] ?? type}
    </span>
  );
}

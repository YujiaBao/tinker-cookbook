/** TypeScript types matching the Tinker Chef backend API responses. */

export interface RunInfo {
  run_id: string;
  path: string;
  has_config: boolean;
  has_metrics: boolean;
  has_checkpoints: boolean;
  has_timing: boolean;
  iteration_count: number;
  status: string;  // "running" | "completed" | "idle"
  last_updated: number | null;  // epoch seconds
  training_type: string | null;  // "rl" | "sl" | "dpo"
  config_summary?: Record<string, unknown>;
  latest_step?: number;
  total_steps?: number;
  config?: Record<string, unknown>;
}

export interface MetricsResponse {
  run_id: string;
  total_records: number;
  records: MetricRecord[];
}

export interface MetricRecord {
  step?: number;
  [key: string]: number | string | undefined;
}

export interface IterationInfo {
  iteration: number;
  has_train_rollouts: boolean;
  has_train_logtree: boolean;
  eval_labels: string[];
}

export interface RolloutSummary {
  group_idx: number;
  traj_idx: number;
  tags: string[];
  total_reward: number;
  final_reward: number;
  num_steps: number;
  final_ob_len: number;
  sampling_client_step?: number;
}

export interface RolloutsResponse {
  run_id: string;
  iteration: number;
  split: string;
  total: number;
  available_tags: string[];
  rollouts: RolloutSummary[];
}

export interface RolloutStep {
  step_idx: number;
  ob_len: number;
  ac_len: number;
  reward: number;
  episode_done: boolean;
  metrics: Record<string, number>;
  logs: Record<string, unknown>;
}

export interface RolloutDetail {
  schema_version: number;
  split: string;
  iteration: number;
  group_idx: number;
  traj_idx: number;
  tags: string[];
  sampling_client_step?: number;
  total_reward: number;
  final_reward: number;
  trajectory_metrics: Record<string, number>;
  steps: RolloutStep[];
  final_ob_len: number;
}

export interface TimingRecord {
  step: number;
  name: string;
  start_time: number;
  end_time: number;
  wall_start: number;
  wall_end: number;
}

export interface TimingResponse {
  run_id: string;
  total_records: number;
  records: TimingRecord[];
}

export interface CheckpointRecord {
  state_path: string;
  name: string;
  kind: string;
  timestamp: number;
  loop_state: { epoch: number; batch: number };
}

export interface LogtreeNode {
  tag: string;
  attrs?: Record<string, string>;
  children?: (string | LogtreeNode)[];
  data?: Record<string, unknown>;
}

export interface LogtreeResponse {
  title: string;
  started_at: string;
  root: LogtreeNode;
}

// Eval benchmark types

export interface EvalRunSummary {
  eval_run_id: string;
  model_name: string;
  checkpoint_path?: string;
  checkpoint_name?: string;
  timestamp?: string;
  benchmarks: string[];
  scores: Record<string, number>;
}

export interface EvalRunDetail {
  eval_run_id: string;
  metadata: Record<string, unknown>;
  benchmarks: string[];
  results: Record<string, EvalBenchmarkResult>;
}

export interface EvalBenchmarkResult {
  name: string;
  score: number;
  num_examples: number;
  num_correct: number;
  num_errors: number;
  metrics: Record<string, number>;
  time_seconds: number;
  pass_at_k?: Record<number, number>;
}

export interface EvalTrajectorySummary {
  idx: number;
  example_id?: string;
  reward: number;
  num_turns: number;
  time_seconds: number;
  error?: string;
  logs: Record<string, unknown>;
}

export interface EvalTrajectoriesResponse {
  eval_run_id: string;
  benchmark: string;
  total: number;
  trajectories: EvalTrajectorySummary[];
}

export interface EvalTrajectoryTurn {
  role: string;
  content: string;
  token_count: number;
  metadata: Record<string, unknown>;
}

export interface EvalTrajectoryDetail {
  idx: number;
  benchmark: string;
  example_id?: string;
  turns: EvalTrajectoryTurn[];
  reward: number;
  metrics: Record<string, number>;
  logs: Record<string, unknown>;
  error?: string;
  time_seconds: number;
}

export interface ScoresTableRow {
  run_id: string;
  model_name: string;
  checkpoint_name?: string;
  timestamp?: string;
  scores: Record<string, number>;
}

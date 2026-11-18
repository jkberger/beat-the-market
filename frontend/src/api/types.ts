export interface Bar {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface IndicatorParam {
  name: string;
  kind: "int" | "float";
  default: number;
  min: number;
  max: number;
}

export interface IndicatorSpec {
  id: string;
  name: string;
  category: "trend" | "momentum" | "volatility" | "volume";
  params: IndicatorParam[];
  outputs: string[];
  overlay: boolean;
}

export interface RangeSpec {
  min: number;
  max: number;
  step: number;
}
export type ParamValue = number | RangeSpec;

export type PriceField = "open" | "high" | "low" | "close";
export type Operand =
  | { kind: "price"; field: PriceField }
  | { kind: "indicator"; id: string; params: Record<string, ParamValue>; output: string }
  | { kind: "constant"; value: ParamValue };

export type Op = ">" | "<" | "crosses_above" | "crosses_below" | "rising" | "falling";

export interface Condition {
  kind: "condition";
  left: Operand;
  op: Op;
  right?: Operand | null;
  bars?: number;
}

export interface Group {
  kind: "group";
  join: "and" | "or";
  items: (Condition | Group)[];
}

export interface StopSpec {
  kind: "percent" | "atr";
  value: ParamValue;
}

export interface Strategy {
  name: string;
  buy: Group;
  sell: Group;
  stop_loss?: StopSpec | null;
  take_profit?: StopSpec | null;
  max_hold_bars?: number | null;
  allow_short: boolean;
}

export type RiskPreset = "conservative" | "balanced" | "aggressive" | "custom";
export interface RiskSettings {
  preset: RiskPreset;
  position_pct: number;
  stop_atr_mult: number | null;
  max_drawdown_pct: number | null;
  standaside_bars: number;
  objective: "sortino" | "sharpe" | "cagr";
}

export interface BacktestConfig {
  initial_equity: number;
  slippage_bps: number;
}

export interface BacktestRequest {
  symbols: string[];
  strategy: Strategy;
  risk: RiskSettings;
  config: BacktestConfig;
  mode: "simple" | "walk_forward";
  start?: string | null;
  end?: string | null;
  score_holdout: boolean;
}

export interface Metrics {
  total_return: number | null;
  cagr: number | null;
  volatility: number | null;
  sharpe: number | null;
  sortino: number | null;
  max_drawdown: number | null;
  calmar: number | null;
  win_rate: number | null;
  avg_win: number | null;
  avg_loss: number | null;
  profit_factor: number | null;
  num_trades: number;
  time_in_market: number | null;
}

export interface Point {
  date: string;
  value: number | null;
}

export interface Trade {
  direction: "long" | "short";
  entry_date: string;
  entry_price: number;
  exit_date: string;
  exit_price: number;
  return_pct: number;
  bars_held: number;
  exit_reason: "signal" | "stop" | "target" | "max_hold" | "drawdown" | "end";
}

export type SegmentLabel = "in_sample" | "out_of_sample" | "holdout" | "backtest";

export interface Segment {
  label: SegmentLabel;
  start: string;
  end: string;
  equity: Point[];
  trades: Trade[];
  metrics: Metrics;
  baseline_metrics: { stock: Metrics | null; spy: Metrics | null };
}

export interface Baseline {
  equity: Point[];
  metrics: Metrics;
}

export interface Marker {
  date: string;
  side: "buy" | "sell";
  price: number;
  reason?: string;
}

export interface WindowInfo {
  window: { train_start: string; train_end: string; test_start: string; test_end: string };
  chosen: Record<string, number>;
  qualified: boolean;
  test: Metrics;
}

export interface Run {
  symbol: string;
  mode: "simple" | "walk_forward";
  strategy_text: string;
  risk: RiskSettings;
  segments: Partial<Record<SegmentLabel, Segment>>;
  baselines: { stock: Baseline | null; spy: Baseline | null };
  markers: Marker[];
  windows: WindowInfo[];
  holdout_count: number | null;
}

export interface BacktestResponse {
  id: number;
  symbol: string;
  mode: string;
  strategy_name: string;
  request: BacktestRequest;
  summary: { median: Metrics; per_symbol: { symbol: string; metrics: Metrics }[] };
  runs: Run[];
}

export interface RunListItem {
  id: number;
  symbol: string;
  mode: string;
  strategy_name: string;
  created_at: string;
  summary: BacktestResponse["summary"] | null;
}

export interface SavedStrategy {
  id: number;
  name: string;
  strategy: Strategy;
  created_at: string;
}

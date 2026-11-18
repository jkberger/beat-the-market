import { Info } from "lucide-react";
import type { Metrics, Segment, SegmentLabel } from "@/api/types";
import { Tooltip } from "@/components/ui/tooltip";
import { formatNumber, formatPct, signClass } from "@/lib/format";
import { METRIC_HELP } from "@/lib/metricsInfo";
import { cn } from "@/lib/utils";

export const SEGMENT_LABEL: Record<SegmentLabel, string> = {
  out_of_sample: "Out-of-sample (walk-forward)",
  holdout: "Holdout (never seen)",
  backtest: "Backtest",
  in_sample: "In-sample (diagnostic)",
};

const SEGMENT_NOTE: Record<SegmentLabel, string> = {
  out_of_sample: "Only the blind 6-month test windows, each traded with settings tuned on the 3 years before it. Tested",
  holdout: "Locked data never used for tuning, scored with the final settings. Covers",
  backtest: "Your fixed rules over the chosen dates. Covers",
  in_sample: "Data the strategy was tuned on. Expect this to look too good. Covers",
};

const ORDER: SegmentLabel[] = ["out_of_sample", "holdout", "backtest", "in_sample"];

function Stat({ k, m, big }: { k: keyof Metrics; m: Metrics; big?: boolean }) {
  const v = m[k];
  const text =
    k === "sharpe" || k === "sortino" || k === "profit_factor"
      ? formatNumber(v as number | null)
      : k === "num_trades"
        ? String(v)
        : formatPct(v as number | null);
  const colored = k === "cagr" || k === "total_return";
  return (
    <div className="flex flex-col">
      <span className="flex items-center gap-1 text-[11px] text-muted">
        {LABELS[k]}
        {METRIC_HELP[k] && (
          <Tooltip text={METRIC_HELP[k]!}>
            <Info className="h-3 w-3" />
          </Tooltip>
        )}
      </span>
      <span data-testid={k} className={cn("num font-semibold", big ? "text-2xl" : "text-sm", colored && signClass(v as number))}>
        {text}
      </span>
    </div>
  );
}

const LABELS: Partial<Record<keyof Metrics, string>> = {
  cagr: "CAGR",
  total_return: "Total return",
  max_drawdown: "Max drawdown",
  sharpe: "Sharpe",
  num_trades: "Trades",
  win_rate: "Win rate",
};

function MetricCard({ title, note, m, accent, muted, footer }: {
  title: string; note?: string; m: Metrics; accent?: boolean; muted?: boolean; footer?: React.ReactNode;
}) {
  return (
    <div data-card className={cn(
      "flex flex-col gap-3 rounded-2xl border bg-surface p-4",
      accent ? "border-accent/50 shadow-[0_0_0_1px] shadow-accent/20" : "border-border",
      muted && "opacity-70",
    )}>
      <div>
        <div className={cn("text-sm font-semibold", accent && "text-accent")}>{title}</div>
        {note && <div className="text-[11px] text-muted">{note}</div>}
      </div>
      <Stat k="cagr" m={m} big />
      <div className="grid grid-cols-2 gap-x-4 gap-y-2">
        <Stat k="total_return" m={m} />
        <Stat k="max_drawdown" m={m} />
        <Stat k="sharpe" m={m} />
        <Stat k="num_trades" m={m} />
      </div>
      {footer}
    </div>
  );
}

function verdict(m: Metrics, base: Metrics | null, symbol: string) {
  if (!base || m.cagr === null || base.cagr === null) return null;
  const diff = m.cagr - base.cagr;
  const pts = `${Math.abs(diff * 100).toFixed(1)} pts a year`;
  return (
    <div className={cn("rounded-lg px-3 py-2 text-xs font-medium", diff >= 0 ? "bg-gain/10 text-gain" : "bg-loss/10 text-loss")}>
      {diff >= 0 ? `Beat buy & hold ${symbol} by ${pts}` : `Trailed buy & hold ${symbol} by ${pts}`}
    </div>
  );
}

interface Props {
  symbol: string;
  segments: Partial<Record<SegmentLabel, Segment>>;
  holdoutCount: number | null;
}

export function MetricsCards({ symbol, segments, holdoutCount }: Props) {
  const present = ORDER.filter((l) => segments[l]);
  const primary = segments[present[0]];
  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      {present.map((label) => {
        const seg = segments[label]!;
        return (
          <MetricCard
            key={label}
            title={SEGMENT_LABEL[label]}
            note={`${SEGMENT_NOTE[label]} ${seg.start} to ${seg.end}.`}
            m={seg.metrics}
            accent={label === "holdout"}
            muted={label === "in_sample"}
            footer={
              <>
                {verdict(seg.metrics, seg.baseline_metrics.stock, symbol)}
                {label === "holdout" && holdoutCount !== null && holdoutCount > 0 && (
                  <div className="text-[11px] text-muted">
                    Scored {holdoutCount} {holdoutCount === 1 ? "time" : "times"}
                    {holdoutCount > 1 && ". Repeated peeks make it less trustworthy."}
                  </div>
                )}
              </>
            }
          />
        );
      })}
      {primary?.baseline_metrics.stock && (
        <MetricCard title={`Buy & hold ${symbol}`} note="Same dates as the first card." m={primary.baseline_metrics.stock} />
      )}
      {primary?.baseline_metrics.spy && symbol !== "SPY" && (
        <MetricCard title="Buy & hold SPY" note="The market, same dates." m={primary.baseline_metrics.spy} />
      )}
    </div>
  );
}

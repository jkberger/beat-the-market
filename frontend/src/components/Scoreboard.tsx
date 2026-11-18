import { Info } from "lucide-react";
import type { Metrics, Segment, SegmentLabel } from "@/api/types";
import { Tooltip } from "@/components/ui/tooltip";
import { formatMoney, formatNumber, formatPct, signClass } from "@/lib/format";
import { cn } from "@/lib/utils";

export const SEGMENT_LABEL: Record<SegmentLabel, string> = {
  out_of_sample: "Blind test",
  holdout: "Final check",
  backtest: "Backtest",
  in_sample: "Training data",
};

export const COLUMN_HELP = {
  perYear: "The average yearly growth (CAGR). 10% means the money grew as if it earned 10% every year.",
  worstDrop:
    "The biggest fall from a high point to a later low (max drawdown). -30% means at its worst the account was 30% below its best value so far.",
  sharpe: "Return for the risk (Sharpe ratio): how much it earned for how bumpy the ride was. Above 1 is good, below 0.5 is weak.",
  trades: "How many times it bought and then sold.",
  winRate: "The share of trades that made money.",
};

function Head({ children, help }: { children: string; help: string }) {
  return (
    <th className="py-2 pl-3 text-right font-medium first:pl-0">
      <span className="inline-flex items-center gap-1">
        {children}
        <Tooltip text={help}>
          <Info className="h-3 w-3" />
        </Tooltip>
      </span>
    </th>
  );
}

interface Row {
  name: string;
  m: Metrics | null;
  strategy?: boolean;
  note?: string;
}

function CompareTable({ rows }: { rows: Row[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm num">
        <thead className="text-xs text-muted">
          <tr className="border-b border-border">
            <th className="w-1/3 py-2 text-left font-medium" />
            <Head help={COLUMN_HELP.perYear}>Per year</Head>
            <Head help={COLUMN_HELP.worstDrop}>Worst drop</Head>
            <Head help={COLUMN_HELP.sharpe}>Return for the risk</Head>
            <Head help={COLUMN_HELP.trades}>Trades</Head>
            <Head help={COLUMN_HELP.winRate}>Winning trades</Head>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.name} data-row={r.name} className={cn("border-b border-border/60 last:border-0", r.strategy && "font-semibold")}>
              <td className="py-2.5 pr-3 text-left">
                {r.name}
                {r.note && <span className="ml-2 text-xs font-normal text-muted">{r.note}</span>}
              </td>
              {r.m ? (
                <>
                  <td className={cn("py-2.5 pl-3 text-right", signClass(r.m.cagr))}>{formatPct(r.m.cagr)}</td>
                  <td className="py-2.5 pl-3 text-right">{formatPct(r.m.max_drawdown)}</td>
                  <td className="py-2.5 pl-3 text-right">{formatNumber(r.m.sharpe)}</td>
                  <td className="py-2.5 pl-3 text-right">{r.strategy ? r.m.num_trades : "—"}</td>
                  <td className="py-2.5 pl-3 text-right">{r.strategy ? formatPct(r.m.win_rate, 0) : "—"}</td>
                </>
              ) : (
                <td colSpan={5} className="py-2.5 pl-3 text-right text-xs text-muted">Not available</td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const grown = (start: number, m: Metrics | null) =>
  m?.total_return == null ? null : start * (1 + m.total_return);

function Headline({ symbol, seg, start }: { symbol: string; seg: Segment; start: number }) {
  const mine = grown(start, seg.metrics);
  const hold = grown(start, seg.baseline_metrics.stock);
  const a = seg.metrics.cagr;
  const b = seg.baseline_metrics.stock?.cagr;
  const beat = a != null && b != null ? a >= b : null;
  const years = new Date(seg.end).getFullYear() - new Date(seg.start).getFullYear();
  return (
    <div className="flex flex-col gap-2">
      {beat !== null && (
        <span className={cn("w-fit rounded-full px-3 py-1 text-xs font-semibold", beat ? "bg-gain/15 text-gain" : "bg-loss/15 text-loss")}>
          {beat ? `Beat just holding ${symbol}` : `Did worse than just holding ${symbol}`}
        </span>
      )}
      <p className="text-lg leading-snug">
        Over {years > 1 ? `${years} years` : "this period"} ({seg.start} to {seg.end}), this strategy turned{" "}
        <span className="num font-semibold">{formatMoney(start)}</span> into{" "}
        <span className={cn("num font-semibold", signClass(seg.metrics.total_return))}>{formatMoney(mine)}</span>.
        {hold !== null && (
          <>
            {" "}Just buying {symbol} and holding it turned it into{" "}
            <span className={cn("num font-semibold", signClass(seg.baseline_metrics.stock?.total_return))}>{formatMoney(hold)}</span>.
          </>
        )}
      </p>
    </div>
  );
}

function rowsFor(symbol: string, seg: Segment): Row[] {
  const rows: Row[] = [
    { name: "This strategy", m: seg.metrics, strategy: true },
    { name: `Just hold ${symbol}`, m: seg.baseline_metrics.stock },
  ];
  if (symbol === "SPY") rows.push({ name: "The market (SPY)", m: seg.baseline_metrics.stock, note: "same as holding SPY" });
  else rows.push({ name: "The market (SPY)", m: seg.baseline_metrics.spy });
  return rows;
}

interface Props {
  symbol: string;
  segments: Partial<Record<SegmentLabel, Segment>>;
  holdoutCount: number | null;
  startingAmount: number;
}

/** The plain-English result for one stock: a headline, then strategy vs holding vs the market. */
export function Scoreboard({ symbol, segments, holdoutCount, startingAmount }: Props) {
  const main = segments.out_of_sample ?? segments.backtest;
  const holdout = segments.holdout;
  return (
    <div className="flex flex-col gap-4 rounded-2xl border border-border bg-surface p-5">
      {main && (
        <>
          <Headline symbol={symbol} seg={main} start={startingAmount} />
          <CompareTable rows={rowsFor(symbol, main)} />
        </>
      )}
      {holdout && (
        <div data-holdout className="flex flex-col gap-2 border-t border-border pt-4">
          <div className="text-sm font-semibold">
            Final check: {holdout.start} to {holdout.end}
            <span className="ml-2 text-xs font-normal text-muted">
              Recent data it never saw, traded once with the final settings.
              {holdoutCount !== null && holdoutCount > 0 &&
                ` Scored ${holdoutCount} ${holdoutCount === 1 ? "time" : "times"}${holdoutCount > 1 ? ", and repeat looks make it less trustworthy" : ""}.`}
            </span>
          </div>
          <CompareTable rows={rowsFor(symbol, holdout)} />
        </div>
      )}
    </div>
  );
}

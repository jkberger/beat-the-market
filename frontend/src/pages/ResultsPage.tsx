import { useQuery } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "@/api/client";
import type { BacktestResponse, Run, SegmentLabel } from "@/api/types";
import { EquityChart, rebase, type EquityLine } from "@/components/EquityChart";
import { ErrorBox, Spinner } from "@/components/ErrorBox";
import { Scoreboard, SEGMENT_LABEL } from "@/components/Scoreboard";
import { PriceChart } from "@/components/PriceChart";
import { TestedDates } from "@/components/TestedDates";
import { TradeTable } from "@/components/TradeTable";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Segmented } from "@/components/ui/segmented";
import { formatPct, signClass } from "@/lib/format";
import { cn } from "@/lib/utils";

const PRESET_NAME = { conservative: "Conservative", balanced: "Balanced", aggressive: "Aggressive", custom: "Custom" };

function AcrossStocks({ data, onPick, active }: { data: BacktestResponse; onPick: (s: string) => void; active: string }) {
  const rows = data.runs.map((run) => {
    const seg = run.segments.out_of_sample ?? run.segments.backtest!;
    const mine = seg.metrics.cagr;
    const hold = seg.baseline_metrics.stock?.cagr ?? null;
    return { symbol: run.symbol, seg, mine, hold, beat: mine != null && hold != null ? mine >= hold : null };
  });
  const wins = rows.filter((r) => r.beat).length;
  return (
    <Card>
      <CardHeader>
        <CardTitle>Did it beat just holding the stock?</CardTitle>
        <CardDescription>
          {wins} of {rows.length} stocks. Pick a stock below for its full story.
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {rows.map((r) => (
          <button key={r.symbol} type="button" onClick={() => onPick(r.symbol)}
            className={cn("flex flex-col gap-1 rounded-xl border p-3 text-left transition-colors hover:bg-surface-2/60",
              r.symbol === active ? "border-accent/60" : "border-border")}>
            <span className="flex items-center justify-between">
              <span className="font-semibold">{r.symbol}</span>
              {r.beat !== null && (
                <span className={cn("rounded-full px-2 py-0.5 text-[11px] font-semibold", r.beat ? "bg-gain/15 text-gain" : "bg-loss/15 text-loss")}>
                  {r.beat ? "Beat holding" : "Worse than holding"}
                </span>
              )}
            </span>
            <span className="num text-sm">
              <span className={signClass(r.mine)}>{formatPct(r.mine)}</span> a year vs{" "}
              <span className={signClass(r.hold)}>{formatPct(r.hold)}</span> holding
            </span>
            <span className="num text-xs text-muted">Worst drop {formatPct(r.seg.metrics.max_drawdown)}</span>
          </button>
        ))}
      </CardContent>
    </Card>
  );
}

function RunView({ run, startingAmount }: { run: Run; startingAmount: number }) {
  const labels = (["out_of_sample", "holdout", "backtest", "in_sample"] as SegmentLabel[]).filter((l) => run.segments[l]);
  const [label, setLabel] = useState<SegmentLabel>(labels[0]);
  const seg = run.segments[label] ?? run.segments[labels[0]]!;
  const barsQ = useQuery({ queryKey: ["bars", run.symbol], queryFn: () => api.bars(run.symbol) });
  const settingsQ = useQuery({ queryKey: ["settings"], queryFn: api.getSettings });
  const holdoutSeg = run.segments.holdout;

  const lines: EquityLine[] = useMemo(() => {
    const first = seg.equity.find((p) => p.value !== null)?.value ?? 10_000;
    const out: EquityLine[] = [{ name: "Strategy", points: seg.equity }];
    if (run.baselines.stock)
      out.push({ name: `Hold ${run.symbol}`, points: rebase(run.baselines.stock.equity, seg.start, seg.end, first), color: "#f59e0b", dashed: true });
    if (run.baselines.spy && run.symbol !== "SPY")
      out.push({ name: "Hold SPY", points: rebase(run.baselines.spy.equity, seg.start, seg.end, first), color: "#06b6d4", dashed: true });
    return out;
  }, [run, seg]);

  const bars = useMemo(
    () => (barsQ.data?.bars ?? []).filter((b) => b.date >= seg.start && b.date <= seg.end),
    [barsQ.data, seg],
  );
  const markers = useMemo(() => run.markers.filter((m) => m.date >= seg.start && m.date <= seg.end), [run, seg]);

  return (
    <div className="flex flex-col gap-6">
      <Scoreboard symbol={run.symbol} segments={run.segments} holdoutCount={run.holdout_count} startingAmount={startingAmount} />
      {run.windows.length > 0 && (
        <TestedDates
          windows={run.windows}
          holdout={holdoutSeg ? { start: holdoutSeg.start, end: holdoutSeg.end } : null}
          holdoutStart={settingsQ.data?.holdout_start}
        />
      )}

      <Card>
        <CardHeader className="flex-row flex-wrap items-center justify-between gap-3">
          <div>
            <CardTitle>How the money grew</CardTitle>
            <CardDescription>The strategy against just holding, starting with the same amount on the same day.</CardDescription>
          </div>
          {labels.length > 1 && (
            <Segmented label="Segment" value={label} onChange={setLabel}
              options={labels.map((l) => ({ value: l, label: SEGMENT_LABEL[l] }))} />
          )}
        </CardHeader>
        <CardContent><EquityChart lines={lines} /></CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Every buy and sell</CardTitle>
          <CardDescription>{markers.filter((m) => m.side === "buy").length} buys, {markers.filter((m) => m.side === "sell").length} sells in this segment.</CardDescription>
        </CardHeader>
        <CardContent>
          {barsQ.isLoading ? <Spinner /> : barsQ.error ? <ErrorBox error={barsQ.error} /> : <PriceChart bars={bars} markers={markers} height={360} />}
        </CardContent>
      </Card>

      {run.windows.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Window by window</CardTitle>
            <CardDescription>For the curious: each 6-month blind test, what it was tuned on, and how it did.</CardDescription>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            <table className="w-full text-sm num">
              <thead className="text-left text-xs text-muted">
                <tr className="border-b border-border">
                  <th className="py-2 font-medium">Trained on</th>
                  <th className="py-2 font-medium">Tested on</th>
                  <th className="py-2 font-medium">Chosen values</th>
                  <th className="py-2 text-right font-medium">Test return</th>
                  <th className="py-2 text-right font-medium">Trades</th>
                </tr>
              </thead>
              <tbody>
                {run.windows.map((w, i) => (
                  <tr key={i} className="border-b border-border/60">
                    <td className="py-1.5">{w.window.train_start} → {w.window.train_end}</td>
                    <td className="py-1.5">{w.window.test_start} → {w.window.test_end}</td>
                    <td className="py-1.5 text-xs text-muted">
                      {Object.keys(w.chosen).length ? Object.values(w.chosen).join(", ") : "fixed rules"}
                      {!w.qualified && Object.keys(w.chosen).length > 0 && <Badge className="ml-2">under 5 trades</Badge>}
                    </td>
                    <td className={cn("py-1.5 text-right", signClass(w.test.total_return))}>{formatPct(w.test.total_return)}</td>
                    <td className="py-1.5 text-right">{w.test.num_trades}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader><CardTitle>Trades</CardTitle></CardHeader>
        <CardContent className="p-0 pb-2"><TradeTable trades={seg.trades} /></CardContent>
      </Card>
    </div>
  );
}

export function ResultsPage() {
  const id = Number(useParams().id);
  const q = useQuery({ queryKey: ["backtest", id], queryFn: () => api.getBacktest(id) });
  const [symbol, setSymbol] = useState<string | null>(null);
  if (q.isLoading) return <Spinner label="Loading results…" />;
  if (q.error) return <ErrorBox error={q.error} />;
  const data = q.data!;
  const run = data.runs.find((r) => r.symbol === symbol) ?? data.runs[0];

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex flex-col gap-2">
          <Link to="/build" className="inline-flex items-center gap-1 text-xs text-muted hover:text-text">
            <ArrowLeft className="h-3 w-3" /> Back to builder
          </Link>
          <h1 className="text-2xl font-semibold tracking-tight">{data.strategy_name}</h1>
          <p className="max-w-3xl text-sm">{run.strategy_text}</p>
          <div className="flex flex-wrap gap-2">
            <Badge>{data.mode === "walk_forward" ? "Tested blind (walk-forward)" : "Simple backtest"}</Badge>
            <Badge>Risk: {PRESET_NAME[run.risk.preset]}</Badge>
            <Badge>{run.risk.position_pct}% per trade</Badge>
            {data.request.config.slippage_bps > 0 && <Badge>{data.request.config.slippage_bps} bps slippage</Badge>}
          </div>
        </div>
        <Link to="/build"><Button variant="secondary">New backtest</Button></Link>
      </div>

      {data.runs.length > 1 && <AcrossStocks data={data} onPick={setSymbol} active={run.symbol} />}
      {data.runs.length > 1 && (
        <Segmented label="Stock" value={run.symbol} onChange={setSymbol}
          options={data.runs.map((r) => ({ value: r.symbol, label: r.symbol }))} />
      )}
      <RunView key={run.symbol} run={run} startingAmount={data.request.config.initial_equity} />
    </div>
  );
}

import { useQuery } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "@/api/client";
import type { BacktestResponse, Run, SegmentLabel } from "@/api/types";
import { EquityChart, rebase, type EquityLine } from "@/components/EquityChart";
import { ErrorBox, Spinner } from "@/components/ErrorBox";
import { MetricsCards, SEGMENT_LABEL } from "@/components/MetricsCards";
import { PriceChart } from "@/components/PriceChart";
import { TradeTable } from "@/components/TradeTable";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Segmented } from "@/components/ui/segmented";
import { formatNumber, formatPct, signClass } from "@/lib/format";
import { cn } from "@/lib/utils";

const PRESET_NAME = { conservative: "Conservative", balanced: "Balanced", aggressive: "Aggressive", custom: "Custom" };

function Comparison({ data, onPick }: { data: BacktestResponse; onPick: (s: string) => void }) {
  const rows = data.summary.per_symbol;
  return (
    <Card>
      <CardHeader>
        <CardTitle>Comparison across stocks</CardTitle>
        <CardDescription>
          {data.mode === "walk_forward" ? "Out-of-sample" : "Backtest"} results per stock. Median CAGR{" "}
          {formatPct(data.summary.median.cagr)}, median Sharpe {formatNumber(data.summary.median.sharpe)}.
        </CardDescription>
      </CardHeader>
      <CardContent className="overflow-x-auto">
        <table className="w-full text-sm num">
          <thead className="text-left text-xs text-muted">
            <tr className="border-b border-border">
              <th className="py-2 font-medium">Stock</th>
              <th className="py-2 text-right font-medium">CAGR</th>
              <th className="py-2 text-right font-medium">vs buy & hold</th>
              <th className="py-2 text-right font-medium">Max drawdown</th>
              <th className="py-2 text-right font-medium">Sharpe</th>
              <th className="py-2 text-right font-medium">Trades</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const run = data.runs.find((x) => x.symbol === r.symbol)!;
              const seg = run.segments.out_of_sample ?? run.segments.backtest!;
              const base = seg.baseline_metrics.stock?.cagr;
              const diff = r.metrics.cagr !== null && base != null ? r.metrics.cagr - base : null;
              return (
                <tr key={r.symbol} className="cursor-pointer border-b border-border/60 hover:bg-surface-2/60" onClick={() => onPick(r.symbol)}>
                  <td className="py-2 font-semibold">{r.symbol}</td>
                  <td className={cn("py-2 text-right", signClass(r.metrics.cagr))}>{formatPct(r.metrics.cagr)}</td>
                  <td className={cn("py-2 text-right", signClass(diff))}>{diff === null ? "—" : `${diff >= 0 ? "+" : ""}${(diff * 100).toFixed(1)} pts`}</td>
                  <td className="py-2 text-right">{formatPct(r.metrics.max_drawdown)}</td>
                  <td className="py-2 text-right">{formatNumber(r.metrics.sharpe)}</td>
                  <td className="py-2 text-right">{r.metrics.num_trades}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}

function RunView({ run }: { run: Run }) {
  const labels = (["out_of_sample", "holdout", "backtest", "in_sample"] as SegmentLabel[]).filter((l) => run.segments[l]);
  const [label, setLabel] = useState<SegmentLabel>(labels[0]);
  const seg = run.segments[label] ?? run.segments[labels[0]]!;
  const barsQ = useQuery({ queryKey: ["bars", run.symbol], queryFn: () => api.bars(run.symbol) });

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
      <MetricsCards symbol={run.symbol} segments={run.segments} holdoutCount={run.holdout_count} />

      <Card>
        <CardHeader className="flex-row flex-wrap items-center justify-between gap-3">
          <div>
            <CardTitle>Equity vs buy & hold</CardTitle>
            <CardDescription>Starting from the same amount on the same day.</CardDescription>
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
            <CardTitle>Walk-forward windows</CardTitle>
            <CardDescription>Each window tuned on its training years, then traded the next 6 months blind.</CardDescription>
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
            <Badge>{data.mode === "walk_forward" ? "Walk-forward" : "Simple backtest"}</Badge>
            <Badge>Risk: {PRESET_NAME[run.risk.preset]}</Badge>
            <Badge>{run.risk.position_pct}% per trade</Badge>
            {data.request.config.slippage_bps > 0 && <Badge>{data.request.config.slippage_bps} bps slippage</Badge>}
          </div>
        </div>
        <Link to="/build"><Button variant="secondary">New backtest</Button></Link>
      </div>

      {data.runs.length > 1 && <Comparison data={data} onPick={setSymbol} />}
      {data.runs.length > 1 && (
        <Segmented label="Stock" value={run.symbol} onChange={setSymbol}
          options={data.runs.map((r) => ({ value: r.symbol, label: r.symbol }))} />
      )}
      <RunView key={run.symbol} run={run} />
    </div>
  );
}
